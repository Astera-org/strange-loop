import torch
from torch.utils.data import Sampler
from torch.utils._pytree import tree_map
from typing import Any


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
	def __init__(self, num_elements: int, seed: int, num_epochs: int=1):
		self.num_epochs = num_epochs
		self.num_elements = num_elements
		self.gen = torch.Generator().manual_seed(seed)

	def __iter__(self):
		for _ in range(self.num_epochs):
			yield from torch.randperm(self.num_elements, generator=self.gen).tolist()

	def __len__(self):
		return self.num_elements


def collate_pytree(items: list[Any]):
	fn = lambda *tens: torch.stack(tens)
	out = tree_map(fn, *items)
	return out




