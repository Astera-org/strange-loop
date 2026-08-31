from functools import partial
from itertools import islice
from typing import Any, Callable, Concatenate, ParamSpec
from jaxtyping import PRNGKeyArray, Array
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import math
from .. import jfuncs

P = ParamSpec("P")

class ShuffleIterator:
	def __init__(
		self, 
		dataset: Any,
		num_elements: int, 
		batch_size: int,
		seed: int, 
		new_epoch_cb: Callable[['ShuffleSampler'], None]=None, 
		num_epochs: int=1
	):
		if num_elements < batch_size:
			raise RuntimeError(
				f"batch size {batch_size} exceeds total dataset size {num_elements}")

		self.ds = dataset
		self.num_elements = num_elements
		self.batch_size = batch_size
		self.fraction = 1.0
		self.epoch = 0 
		self.step_idx = 0
		self.new_epoch_cb = new_epoch_cb
		self.num_epochs = num_epochs
		self.steps_per_epoch = self.num_elements // self.batch_size
		self.total_steps = self.steps_per_epoch * self.num_epochs
		self.key = jax.random.key(seed) # constant for the life of ShuffleIterator

	@property
	def sampled_size(self):
		return math.ceil(self.num_elements * self.fraction)

	"""
	def index_gen(self):
		for e in range(self.num_epochs):
			epoch_key = jax.random.fold_in(self.key, e)
			perm = np.asarray(jax.random.permutation(epoch_key, self.sampled_size))
			yield from perm
			self.epoch += 1
			if self.new_epoch_cb is not None:
				self.new_epoch_cb(self)
	"""

	@eqx.filter_jit
	def _step(self, key: PRNGKeyArray, step: Array):
		epoch = step // self.steps_per_epoch
		batch_idx = step % self.steps_per_epoch
		epoch_key = jax.random.fold_in(key, epoch)
		offset = batch_idx * self.batch_size
		inds = jfuncs.permute_range(epoch_key, self.sampled_size, self.batch_size, 4, offset)
		key_B = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, inds)
		return self.ds._gen_item(key_B)

	def __iter__(self):
		self.step_idx = 0 
		return self

	def __next__(self):
		if self.step_idx >= self.total_steps:
			raise StopIteration
		item = self._step(self.key, jnp.array(self.step_idx))
		self.step_idx += 1 
		return item

	def __len__(self):
		return self.num_elements

	def set_dataset_fraction(self, fraction: float) -> None:
		if not 0 < fraction <= 1.0:
			raise RuntimeError(f"fraction must be in (0, 1].  Got {fraction}")
		self.fraction = fraction

	def mapreduce(
		self,
		map_fn: Callable,
		reduce_fn: Callable, 
		init: jax.Array,
		map_kwargs=None,
		reduce_kwargs=None,
	) -> jax.Array:
		"""
		Perform a mapreduce across the dataset.

		Inputs:
		  map_fn: item, **map_kwargs -> result
		  reduce_fn: accu, result, **reduce_kwargs -> accu
		  init: initial accumulator value.  Must be such that reduce_fn(init, X) == X

		Outputs:
		  accu: jax.Array.  equivalent to:
			accu = init
			for batch_idx, item in ds:
			  result = map_fn(item, batch_idx)
			  accu = reduce_fn(
			  accu = fold_fn(accu, item, batch_idx)
			return accu
		"""
		if map_kwargs is None:
			map_kwargs = {}
		if reduce_kwargs is None:
			reduce_kwargs = {}

		wrap_map_fn = partial(map_fn, **map_kwargs)
		wrap_reduce_fn = partial(reduce_fn, **reduce_kwargs)

		B = self.batch_size

		def reduce_batch(start):
			idxs = start + jnp.arange(B)
			key_B = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, idxs)
			item = self.ds._gen_item(key_B)
			res_B = jax.vmap(wrap_map_fn)(item)
			return jax.lax.reduce(res_B, init, wrap_reduce_fn, dimensions=(0,))

		def body(carry, offset):
			return reduce_fn(carry, reduce_batch(offset)), None

		starts = jnp.arange(0, self.num_elements, self.batch_size)
		accu, _ = jax.lax.scan(body, init, starts)
		return accu

	def mapreduce_torch(
		self,
		batch_map_fn: Callable, # jax.Array -> jax.Array batched with torch intermediates.
		reduce_fn: Callable,
		accu: jax.Array,
		map_kwargs=None,
		reduce_kwargs=None,
	) -> jax.Array:
		"""
		Like mapreduce, except that batch_map_fn has torch intermediate and is
		batched.

		batch_map_fn: item, **map_kwargs -> result_B
		reduce_fn: pytree of functions which is a prefix of result
		    functions should have signature:
		    accu_leaf, result_leaf, **reduce_kwargs -> accu_leaf
		accu: pytree with same structure as result.
		      accu must satisfy reduce_fn(accu, result) == result
		"""
		if map_kwargs is None:
			map_kwargs = {}
		if reduce_kwargs is None:
			reduce_kwargs = {}

		wrap_batch_map_fn = partial(batch_map_fn, **map_kwargs)
		wrap_reduce_fn = partial(reduce_fn, **reduce_kwargs)

		@jax.jit
		def gen_batch(start):
			idxs = start + jnp.arange(B)
			key_B = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, idxs)
			return self.ds._gen_item(key_B)

		B = self.batch_size

		reduce_init = jax.tree.map(lambda x: x.flatten()[0], accu) 
		reduce_fns = jax.tree.broadcast(wrap_reduce_fn, reduce_init, is_leaf=callable)

		for start in range(0, self.num_elements, B):
			item_B = gen_batch(start)
			res_B = wrap_batch_map_fn(item_B)
			res = jax.tree.map(
					lambda fn, x, r: jax.lax.reduce(x, r, fn, dimensions=(0,)),
					reduce_fns, res_B, reduce_init,
					is_leaf=callable
			)
			accu = jax.tree.map(lambda fn, a, r: fn(a, r), reduce_fns, accu, res)
		return accu

