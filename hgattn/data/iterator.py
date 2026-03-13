from itertools import islice
from typing import Any, Callable
import jax
import jax.numpy as jnp
import math


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
		dataset: Any,
		num_elements: int, 
		batch_size: int,
		seed: int, 
		new_epoch_cb: Callable[['ShuffleSampler'], None]=None, 
		num_epochs: int=1
	):
		self.ds = dataset
		self.num_elements = num_elements
		self.batch_size = batch_size
		self.fraction = 1.0
		self.epoch = 0
		self.new_epoch_cb = new_epoch_cb
		self.num_epochs = num_epochs
		self.key = jax.random.key(seed)
		self.gen = self.index_gen()

	def index_gen(self):
		for e in range(self.num_epochs):
			self.key = jax.random.fold_in(self.key, e)
			yield from jax.random.permutation(self.key, self.sampled_size)
			self.epoch += 1
			if self.new_epoch_cb is not None:
				self.new_epoch_cb(self)

	def __iter__(self):
		return self

	def __next__(self):
		inds = jnp.array(list(islice(self.gen, self.batch_size)))
		if inds.shape[0] != self.batch_size:
			raise StopIteration

		key_B = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(self.key, inds)
		return self.ds._gen_item(key_B)

	def __len__(self):
		return self.num_elements

	def set_dataset_fraction(self, fraction: float) -> None:
		if not 0 < fraction <= 1.0:
			raise RuntimeError(f"fraction must be in (0, 1].  Got {fraction}")
		self.fraction = fraction

	@property
	def sampled_size(self):
		return math.ceil(self.num_elements * self.fraction)




