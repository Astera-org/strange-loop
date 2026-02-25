from dataclasses import dataclass, asdict, field
import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from music21 import converter
import json

"""
Encodes a collection of melodies from https://abc.sourceforge.net/
(folk tunes of Western European Origin which can be written in one staff)

https://abc.sourceforge.net/abcMIDI/original/abcguide.txt
"""

@dataclass
class MelodyDataOpts:
	data_dir: str
	json_file: str
	ctx_len: int
	use_cls_token: bool
	output_onehot: bool
	batch_size: int
	num_tempos: int
	num_tempos_in_train: int


@dataclass
class Melody:
	notes: np.ndarray 
	durations: tuple[float]
	starts: np.ndarray = field(init=False)

	@classmethod
	def from_dict(cls, data):
		return cls(
				notes=np.array(data["notes"]),
				durations=tuple(data["durations"])
				)

	def to_dict(self):
		obj = {
				"notes": self.notes.tolist(),
				"durations": tuple(self.durations)
				}
		return obj

	def __post_init__(self):
		self.starts = np.cumsum((0,) + self.durations)

class MelodyFactory:

	def __init__(self):
		self.titles = {} # score_id => title 
		self.output_classes = {} # score_id => output_class
		self.melodies = {} # score_id => Melody 
		self.tokens = {} # pitch => token (or 'PAD' or 'CLS')
		self.initialized = False

	@property
	def num_classes(self):
		return len(self.output_classes)

	@property
	def num_tokens(self):
		return len(self.tokens)

	def parse(self, abc_dir: str):
		scores = []
		for root, _, files in os.walk(abc_dir): 
			top = Path(root)
			for file in files:
				print(f"parsing {top}/{file}")
				with open(top / file, "r") as fh:
					abc_content = fh.read()
					try:
						opus = converter.parse(abc_content, format="abc")
					except Exception as ex:
						print(f"exception parsing {file}: {ex}. Skipping")
						continue
					scores.extend(list(opus.scores))

		def _extract(note):
			if note.isNote:
				return note.pitch.midi, note.duration.quarterLength
			elif note.isRest:
				return "Rest", note.duration.quarterLength
			else:
				pass

		all_pitches = set()
		items_tmp = {} # score_id => pitches, durations
		for score in scores:
			meta = score.metadata
			pitches = []
			durations = []
			for note in score.flatten().notesAndRests:
				duration = float(note.duration.quarterLength)
				if note.isNote:
					pitch = note.pitch.midi
					all_pitches.add(pitch)
				elif note.isRest:
					pitch = 'Rest'
				else:
					pitch = None

				if pitch is not None:
					pitches.append(pitch)
					durations.append(duration)

			# hack - filter out too-short melodies
			if len(pitches) < 5:
				continue

			items_tmp[meta.id] = pitches, durations
			self.titles[meta.id] = meta.title

		all_pitches_sorted = list(sorted(all_pitches)) + ['Rest', 'PAD', 'CLS']
		self.tokens = { pitch: tok for tok, pitch in enumerate(all_pitches_sorted) }

		# second pass using tokens
		for score_id, (pitches, durations) in items_tmp.items():
			tokens = tuple(self.tokens[p] for p in pitches)
			melody = Melody(np.array(tokens), tuple(durations))
			self.melodies[score_id] = melody

		self.output_classes = { s: i for i, s in enumerate(self.titles.keys()) }
		self.initialized = True

	def save(self, path: str):
		with open(path, "w") as fh:
			melodies = { k: v.to_dict() for k, v in self.melodies.items() }
			obj = {
					"titles": self.titles,
					"tokens": self.tokens,
					"melodies": melodies,
					}
			content = json.dump(obj, fh)

	def load(self, path: str):
		with open(path, "r") as fh:
			items = json.load(fh)
			self.titles = items["titles"]
			self.output_classes = { s: i for i, s in enumerate(self.titles.keys()) }
			self.melodies = { k: Melody.from_dict(v) for k, v in items["melodies"].items() } 
			self.tokens = items["tokens"]
		self.initialized = True

	def get_datasets(
			self, 
			ctx_len: int,
			use_cls_token: bool,
			output_onehot: bool,
			num_tempos: int,
			num_tempos_in_train: int,
			) -> tuple['MelodyDataset', 'MelodyDataset']:
		"""
		Produce a train + test dataset pair from a total set consisting of each
		melody played at `num_tempos` tempos.  The split ensures that
		`num_tempos_in_train` of each melody appear in train, and the rest in test.

		Tempos are evenly dispersed so the melody takes up from 50% and 100% of
		the `ctx_len`.
		"""
		if not 0 < num_tempos_in_train < num_tempos:
			raise RuntimeError(
					f"num_tempos_in_train = {num_tempos_in_train} but must be "
					f"in (0, num_tempos), with num_tempos = {num_tempos}")

		if not self.initialized:
			raise RuntimeError(f"factory is not initialized.  Call parse() or load() first")

		ctx_fractions = np.linspace(0.5, 0.99, num_tempos)
		train_samples = []
		test_samples = []
		for score_id in self.titles.keys():
			np.random.shuffle(ctx_fractions)
			train_tempi, test_tempi = np.split(ctx_fractions, (num_tempos_in_train,))
			train_samples.extend([(score_id, t) for t in train_tempi])
			test_samples.extend([(score_id, t) for t in test_tempi])

		train = MelodyDataset(ctx_len, use_cls_token, output_onehot, self, train_samples)
		test = MelodyDataset(ctx_len, use_cls_token, output_onehot, self, test_samples)
		return train, test

	def _play(
			self, 
			ctx_len: int, 
			score_id: int, 
			ctx_fraction: float,
			) -> np.ndarray:
		"""
		play melody identified by score_id, using ctx_len tokens.
		play the melody at a tempo such that it takes up `ctx_fraction` fraction of
		the total context length.
		"""
		if not (0.25 <= ctx_fraction <= 1.0):
			raise RuntimeError(f"ctx_fraction must be in [0.25, 1],  received {ctx_fraction}")

		m = self.melodies.get(score_id)
		if m is None:
			raise RuntimeError(f"Couldn't find melody identified by score_id {score_id}")

		num_midpoints = int(ctx_len * ctx_fraction) - 1 # allow room for possible cls token
		total_time = m.starts[-1]
		half_token_dur = (total_time / num_midpoints / 2)
		start, stop = half_token_dur, total_time - half_token_dur 
		midpoints = np.linspace(start, stop, num_midpoints)
		inds = np.searchsorted(m.starts, midpoints, side='right') - 1
		return m.notes[inds]


class MelodyDataset(Dataset):

	def __init__(
			self, 
			ctx_len: int,
			use_cls_token: bool,
			output_onehot: bool,
			data: MelodyFactory,
			samples: list[tuple[int, float]], # list of (score_id, ctx_fraction)
			):
		"""
		Defines a set of samples from the data factory.
		samples[i] = (score_id, ctx_fraction), where ctx_fraction is in [0.5, 1] and
		represents the fraction of ctx_len taken up by the melody
		"""
		self.ctx_len = ctx_len
		self.use_cls_token = use_cls_token
		self.output_onehot = output_onehot
		self.data = data
		self.samples = samples

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index: int):
		score_id, ctx_frac = self.samples[index]
		melody = self.data.melodies[score_id]
		notes = self.data._play(self.ctx_len, score_id, ctx_frac)
		notes = torch.tensor(notes, dtype=torch.long)
		tokens = torch.full((self.ctx_len,), self.data.tokens['PAD'])
		if self.use_cls_token:
			tokens[0] = self.data.tokens['CLS']
			tokens[1:notes.shape[0]+1] = notes
		else:
			tokens[:notes.shape[0]] = notes

		output_class = torch.tensor(self.data.output_classes[score_id])
		pad_mask = (tokens != self.data.tokens['PAD']).to(torch.bool)

		out = {
				"score-id": torch.tensor(int(score_id)),
				"output-class": output_class,
				"ctx-fraction": torch.tensor(ctx_frac),
				"notes-ids": tokens,
				"pad-mask": pad_mask
				}
		if self.output_onehot:
			out["notes"] = F.one_hot(tokens, num_classes=self.data.num_tokens)

		return out
