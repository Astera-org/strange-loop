import torch
from torch.utils.data import Sampler
from torch.utils._pytree import tree_map
from typing import Any, Callable
import math


class LoopedRandomSampler(Sampler):
	
	def __init__(self, num_elements: int, seed: int, num_epochs: int=1):
		self.num_epochs = num_epochs
		self.num_elements = num_elements
		self.gen = torch.Generator().manual_seed(seed)

	def __iter__(self):
		for _ in range(self.num_epochs):
			yield from torch.randperm(self.num_elements, generator=self.gen).tolist()

	def __len__(self):
		return 2**64


class ShuffleSampler(Sampler):
	def __init__(
		self, 
		num_elements: int, 
		seed: int, 
		new_epoch_cb: Callable[['ShuffleSampler'], None]=None, 
		num_epochs: int=1
	):
		self.num_epochs = num_epochs
		self.num_elements = num_elements
		self.fraction = 1.0
		self.gen = torch.Generator().manual_seed(seed)
		self.epoch = 0
		self.new_epoch_cb = new_epoch_cb

	def __iter__(self):
		for _ in range(self.num_epochs):
			yield from torch.randperm(self.sampled_size, generator=self.gen).tolist()
			self.epoch += 1
			if self.new_epoch_cb is not None:
				self.new_epoch_cb(self)

	def __len__(self):
		return self.num_elements

	def set_dataset_fraction(self, fraction: float) -> None:
		if not 0 < fraction <= 1.0:
			raise RuntimeError(f"fraction must be in (0, 1].  Got {fraction}")
		self.fraction = fraction

	@property
	def sampled_size(self):
		return math.ceil(self.num_elements * self.fraction)


def collate_pytree(items: list[Any]):
	fn = lambda *tens: torch.stack(tens)
	out = tree_map(fn, *items)
	return out




