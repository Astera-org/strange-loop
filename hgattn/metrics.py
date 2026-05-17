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

@dataclass
class MetricOpts:
	active: bool
	metric_name: str  # name of the metric to set the schedule 
	step_interval: float  # quantity of metric movement triggering a logging event
	target_cats: tuple[TargetCategory]
	num_samples: int
	batch_size: int

	def __post_init__(self):
		try:
			self.target_cats = tuple(TargetCategory(tm) for tm in self.target_cats)
		except Exception as ex:
			raise RuntimeError(
					f"One of target_cats invalid.  Must be one of "
					f"{', '.join(TargetCategory)}")

def granular_metrics(
		dataset: InductiveDataset,
		model: GenerativeModel,
		target_cats: list[TargetCategory],
		num_samples: int,
		batch_size: int,
		seed: int,
		) -> dict[str, jax.Array]:
	"""
	Compute granular metrics defined by `model` induced by `num_samples` from the
	`dataset.  Use `batch_size` for processing, and `target_cat` to define how to
	assign tokens into buckets.
	"""
	def torch_to_jax(ten):
		return jax.dlpack.from_dlpack(ten.contiguous())

	def convert_metric(metric, target_cat_C, *, category):
		buf = dataset.get_target_init(category)
		return buf.at[target_cat_C].set(metric)

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
		cats = { cat.value: dataset.get_target_cat(target_code_BC, cat) for cat in target_cats }
		funs = { cat.value: partial(convert_metric, category=cat) for cat in target_cats }
		return {
			mname: { 
				cname: jax.vmap(funs[cname])(metric, target_cat_BC)
				for cname, target_cat_BC in cats.items()
			} for mname, metric in metrics.items()
		}

	def reduce_fn(accu, result):
		return jax.tree.map(jnp.add, accu, result)

	it = ShuffleIterator(dataset, num_samples, batch_size, seed) 

	init = {
			mname: {
				cat.value: dataset.get_target_init(cat)
				for cat in target_cats
				} for mname in model.metrics_keys()
			}

	return it.mapreduce_torch(batch_map_fn, reduce_fn, init) 

