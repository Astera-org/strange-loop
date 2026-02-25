import torch
from torch.utils.data import Sampler


class LoopedRandomSampler(Sampler):
	
	def __init__(self, num_elements: int):
		self.num_elements = num_elements

	def __iter__(self):
		while True:
			yield from torch.randperm(self.num_elements).tolist()

	def __len__(self):
		return 2**64

