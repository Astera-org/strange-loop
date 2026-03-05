import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from .types import TokensAndProbs

"""
A dataset with a 'copy-offset' operation, interspersed with random numbers.

If CP occurs at position t, then token[t+1] = token[t-1-token[t-1]]

Alphabet is 0 - K, CP
"""

@dataclass
class CopyOffsetOpts:
	context_len: int
	num_vals: int
	op_frequency: float
	dataset_size: int
	seed: int


class CopyOffsetDataset(Dataset):
	def __init__(
		self,
		context_len: int,
		num_vals: int,
		op_frequency: float,
		dataset_size: int,
		seed: int,
	):
		self.context_len = context_len
		self.dataset_size = dataset_size
		self.num_vals = num_vals
		self.gen = torch.Generator()
		self.seed = seed

		if not (0.0 < op_frequency < 0.2):
			raise RuntimeError(f"op_frequency must be in (0, 0.2), received {op_frequency}")
		self.op_frequency = op_frequency

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index: int):
		self.gen.manual_seed(self.seed + index)
		C = self.context_len
		V = self.num_vals + 1
		optoken = V - 1
		inds = torch.arange(C)
		base = torch.randint(0, V-1, (C,), generator=self.gen)

		# target_mask[t] == True means there is a CP token at position t
		target_mask = torch.randint(0, int(1 / self.op_frequency), (C,), generator=self.gen) == 0
		target_mask[:V] = False # prevent OP too early
		target_mask[C-2:] = False
		ops_mask = torch.full((C,), False)
		ops_mask[:-1] = target_mask[1:]

		source_inds = torch.where(target_mask, inds - base[inds - 2] - 2, inds)
		vals = base[source_inds]
		vals = torch.where(ops_mask, optoken, vals)

		return TokensAndProbs(obs_sym=vals, obs_prob=...)

	@property
	def vocab_size(self):
		return self.num_vals + 1 # values + OP

