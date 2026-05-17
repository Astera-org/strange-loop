from functools import partial
from itertools import islice
from typing import Any, Callable, Concatenate, ParamSpec
import jax
import jax.numpy as jnp
import equinox as eqx
import math

P = ParamSpec("P")

"""
class LoopedRandomIterator:
	
	def __init__(self, num_elements: int, seed: int, num_epochs: int=1):
		self.num_epochs = num_epochs
		self.num_elements = num_elements
		self.gen = torch.Generator().manual_seed(seed)

	def __iter__(self):
		for _ in range(self.num_epochs):
			yield from torch.randperm(self.num_elements, generator=self.gen).tolist()

	def __len__(self):
		return 2**64
"""

class ShuffleIterator:
	def __init__(
		self,
	):
		pass


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
		self.new_epoch_cb = new_epoch_cb
		self.num_epochs = num_epochs
		self.key = jax.random.key(seed) # constant for the life of ShuffleIterator
		self.gen = self.index_gen()

	@property
	def sampled_size(self):
		return math.ceil(self.num_elements * self.fraction)

	def index_gen(self):
		for e in range(self.num_epochs):
			epoch_key = jax.random.fold_in(self.key, e)
			yield from jax.random.permutation(epoch_key, self.sampled_size)
			self.epoch += 1
			if self.new_epoch_cb is not None:
				self.new_epoch_cb(self)

	def __iter__(self):
		return self

	def __next__(self):
		inds = jnp.array(list(islice(self.gen, self.batch_size)))
		if inds.shape[0] != self.batch_size:
			raise StopIteration
		# print(f"epoch: {self.epoch}: inds[:10]: {inds[:10]}")

		key_B = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, inds)
		return self.ds._gen_item(key_B)

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

		batch_map_fn: item, **map_kwargs -> result
		reduce_fn: accu, result, **reduce_kwargs -> accu
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
		for start in range(0, self.num_elements, B):
			item_B = gen_batch(start)
			res_B = wrap_batch_map_fn(item_B)
			res = jax.lax.reduce(res_B, reduce_init, wrap_reduce_fn, dimensions=(0,))
			accu = wrap_reduce_fn(accu, res)
		return accu

