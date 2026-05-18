"""
File for implementing analyses that combine a model with a dataset
"""
from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
import jax
import torch
import equinox as eqx
from .data.iterator import ShuffleIterator
from .data.types import TokensAndProbs 
from .data.expression import TargetCategory, InductiveDataset
from .models.generative import GenerativeModel
from . import jfuncs

@dataclass
class MetricOpts:
	active: bool
	step_interval: float  # quantity of metric movement triggering a logging event
	target_cats: tuple[TargetCategory]
	splits: tuple[str] # which data splits to run, e.g. ['train', 'test']
	num_samples: int
	batch_size: int

	def __post_init__(self):
		try:
			self.target_cats = tuple(TargetCategory(tm) for tm in self.target_cats)
		except Exception as ex:
			raise RuntimeError(
					f"One of target_cats invalid.  Must be one of "
					f"{', '.join(TargetCategory)}")
		if not all(s in ('train', 'test') for s in self.splits):
			raise RuntimeError(
					f"splits must be one or more of 'train' or 'test'. "
					f"Got {self.splits}")

def granular_metrics(
		dataset: InductiveDataset,
		model: GenerativeModel,
		target_cats: list[TargetCategory],
		num_samples: int,
		batch_size: int,
		seed: int,
		) -> tuple:
	"""
	Compute granular metrics defined by `model` induced by `num_samples` from the
	`dataset.  Use `batch_size` for processing, and `target_cat` to define how to
	assign tokens into buckets.

	Returns a tuple of:
	  sums[cat][metric] = jax.Array
	  counts[cat] = jax.Array
	  labels[cat] = np.array
	"""
	def torch_to_jax(ten):
		return jax.dlpack.from_dlpack(ten.contiguous())

	def convert_metric(metric, target_cat_C, *, category):
		buf = dataset.get_target_init(category)
		return buf.at[target_cat_C].add(metric)

	def batch_map_fn(
		item: TokensAndProbs,
	) -> tuple[dict, jax.Array]:
		item = item.to_torch()
		inputs = model.prepare_inputs(item, False)
		metrics = model.granular_metrics(
			inputs.input_BC, inputs.input_mask_BC, inputs.label_BC, inputs.label_prob_BCV
		)
		target_code_BC = torch_to_jax(inputs.target_code_BC)
		metrics = jax.tree.map(torch_to_jax, metrics)
		ones = jnp.ones_like(target_code_BC)

		sums = {
			cat: { 
				mname: jax.vmap(partial(convert_metric, category=cat))(
					metric, dataset.get_target_cat(target_code_BC, cat)
				)
		        for mname, metric in metrics.items()
		    } for cat in target_cats
		}
		counts = { 
			cat: jax.vmap(partial(convert_metric, category=cat))(
				ones, dataset.get_target_cat(target_code_BC, cat)
			) for cat in target_cats
		}
		return sums, counts

	it = ShuffleIterator(dataset, num_samples, batch_size, seed) 

	init_sums = {
		cat: { mname: dataset.get_target_init(cat) for mname in model.metrics_keys() } 
		for cat in target_cats
	}
	init_counts = { cat: dataset.get_target_init(cat) for cat in target_cats }
	init = init_sums, init_counts
	sums, counts = it.mapreduce_torch(batch_map_fn, jnp.add, init) 
	labels = { cat: dataset.get_target_label(cat) for cat in target_cats }
	masks = jax.tree.map(lambda c: c != 0, counts)

	def trim(mask, vals):
		vals, ct = jfuncs.compact_masked(vals, mask)
		return vals[:ct]

	def nptrim(mask, vals):
		return vals[mask]

	def sum_trim(mask, msums):
		return jax.tree.map(partial(trim, mask), msums) 

	sums = jax.tree.map(sum_trim, masks, sums)
	counts = jax.tree.map(trim, masks, counts)
	labels = jax.tree.map(nptrim, masks, labels)

	return sums, counts, labels

