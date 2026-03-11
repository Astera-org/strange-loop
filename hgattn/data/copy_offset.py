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
	only_copy_active: bool # if True, only the copied token is active in the mask


class CopyOffsetDataset(Dataset):
	def __init__(self, opts: CopyOffsetOpts, rand_seed: int):

		self.context_len = opts.context_len
		self.dataset_size = opts.dataset_size
		self.num_vals = opts.num_vals
		# self.gen = torch.Generator()
		self.seed = rand_seed 
		self.only_copy_active = opts.only_copy_active

		if not (0.0 < opts.op_frequency < 0.2):
			raise RuntimeError(f"op_frequency must be in (0, 0.2), received {opts.op_frequency}")
		self.op_frequency = opts.op_frequency

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index: int):
		gen = torch.Generator()
		gen.manual_seed(hash((self.seed, index)) % (2**32))
		# self.gen.manual_seed(self.seed + index)
		C = self.context_len
		V = self.num_vals + 1
		optoken = V - 1
		inds = torch.arange(C)
		base = torch.randint(0, V-1, (C,), generator=gen)

		# dest_mask has positions just after all CP tokens
		dest_mask = torch.randint(0, int(1 / self.op_frequency), (C,), generator=gen) == 0
		dest_mask[:V] = False # prevent OP too early
		dest_mask[C-2:] = False
		ops_mask = torch.full((C,), False)
		ops_mask[:-1] = dest_mask[1:] # positions of CP tokens 

		source_inds = torch.where(dest_mask, inds - base[inds - 2] - 2, inds)
		vals = base[source_inds]
		vals = torch.where(ops_mask, optoken, vals)
		one_hot = F.one_hot(vals, num_classes=V).to(torch.float32)
		obs_prob = torch.where(dest_mask[:,None], one_hot, (V ** -1))
		input_mask = torch.full(vals.shape, True)

		if self.only_copy_active:
			target_mask = dest_mask 
		else:
			target_mask = input_mask

		return TokensAndProbs(
			obs_sym=vals, obs_prob=obs_prob, input_mask=input_mask, target_mask=target_mask
		)

	@property
	def loss_label_mask(self):
		return 'copy_tokens_only' if self.only_copy_active else 'all_tokens'

	@property
	def vocab_size(self):
		return self.num_vals + 1 # values + OP
