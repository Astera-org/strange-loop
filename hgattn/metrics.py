"""
File for implementing analyses that combine a model with a dataset
"""
import jax.numpy as jnp
import jax
from .data.iterator import ShuffleIterator
from .data import expression
from .data.types import TokensAndProbs 

@eqx.filter_jit
def granular_metrics(
	dataset: InductiveDataset,
	model: GenerativeModel,
	target_mode: expression.TargetCategory,
	num_samples: int,
	batch_size: int,
	seed: int,
) -> dict[str, jax.Array]:
	"""
	Compute granular metrics defined by `model` induced by `num_samples` from the
	`dataset.  Use `batch_size` for processing, and `target_mode` to define how to
	assign tokens into buckets.
	"""
	def torch_to_jax(ten):
		return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(ten))

    # fold_fn: accu, item, batch_idx, **kwargs -> accu
	def fold_fn(
		accu: jax.Array,
		item: TokensAndProbs,
		batch_idx: int,
		**kwargs,
	) -> jax.Array:
		target_cat = item.target_cat
		item = item.to_torch()
		inputs = model.item_input(item)
		metrics = model.granular_metrics(**inputs)
		metrics = jax.tree.map(torch_to_jax, metrics)
		metric_cat = dataset.get_target_cat(target_mode, target_cat)

		def reduce_fn(accu, metric):
			return accu.at[metric_cat].add(metric)

		return jax.tree.map(reduce_fn, accu, metrics)
	
	it = ShuffleIterator(dataset, num_elements, batch_size, seed) 

	# need a properly shaped initial accu
	initial_accu = dataset.get_target_init(target_mode)
	return it.fold(fold_fn, initial_accu)

