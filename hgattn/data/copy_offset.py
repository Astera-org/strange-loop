import torch
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from .types import TokensAndProbs

"""
A dataset with a 'copy-offset' operation, interspersed with random numbers.

Synopsis:

The alphabet consists of [0, 1, 2, ..., V-1, CP]
The realizations are just random draws from this alphabet, except for one rule:

If ctx[t] = CP, then ctx[t+1] = ctx[t-1-offset], where offset = ctx[t-1].  For example:

0  4  3  2  CP  4  ...

ctx[4] = CP
offset = ctx[3] = 2
ctx[3-offset] = ctx[3-2] = 4

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
		one_hot = F.one_hot(vals, num_classes=V).to(torch.float32)
		obs_prob = torch.where(target_mask[:,None], one_hot, (V ** -1))
		obs_mask = torch.full(vals.shape, True)
		return TokensAndProbs(obs_sym=vals, obs_prob=obs_prob, obs_mask=obs_mask)

	@property
	def vocab_size(self):
		return self.num_vals + 1 # values + OP
