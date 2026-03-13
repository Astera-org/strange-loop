import math
import torch
from torch.utils.data import Dataset
from torch._higher_order_ops import scan
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from .types import TokensAndProbs
from .. import funcs

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
	fixed_offsets: list[int]|None # 

	def __post_init__(self):
		if self.fixed_offsets is None:
			self.fixed_offsets = list(range(self.num_vals))

		if not all(0 <= f < self.num_vals for f in self.fixed_offsets):
			raise RuntimeError(
				f"fixed offsets must all be in [0, num_vals). "
				f"Got {self.fixed_offsets} for num_vals={self.num_vals}")


class CopyOffsetDataset(Dataset):
	def __init__(self, opts: CopyOffsetOpts, rand_seed: int):

		self.context_len = opts.context_len
		self.dataset_size = opts.dataset_size
		self.num_vals = opts.num_vals
		self.op_token = opts.num_vals
		self.seed = rand_seed 
		self.only_copy_active = opts.only_copy_active
		self.fixed_offsets = torch.tensor(opts.fixed_offsets)
		self.max_fixed_offset = max(opts.fixed_offsets)

		if not (0.0 < opts.op_frequency < 0.2):
			raise RuntimeError(f"op_frequency must be in (0, 0.2), received {opts.op_frequency}")
		self.op_frequency = opts.op_frequency
		self.inv_op_frequency = math.ceil(1.0 / self.op_frequency)

	def to(self, device):
		self.fixed_offsets = self.fixed_offsets.to(device)


	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index: int):
		gen = torch.Generator(device=self.fixed_offsets.device)
		gen.manual_seed(hash((self.seed, index)) % (2**32))
		C = self.context_len
		V = self.num_vals + 1

		def scan_fn(state: Tensor, _) -> tuple[Tensor, Tensor]:
			"""
			state is a Tensor of [2, L] where state[0,i] is a token to be written
			in state[1,i] steps from now
			"""
			token = torch.tensor(0, device=state.device) 

			slot = funcs.find_first_value(state[1], 0)
			token = torch.where(
				slot >= 0, 
				state[0, slot], 
				torch.randint(0, V, (), generator=gen, device=state.device)
			)
			is_source = torch.randint(
				0, self.inv_op_frequency, (), generator=gen, device=state.device) == 0
			is_source = torch.logical_and(is_source, slot == -1)

			chosen = torch.randint(
				0, self.fixed_offsets.numel(), (), generator=gen, device=state.device)
			offset = self.fixed_offsets[chosen]
			new_slots = torch.arange(offset, offset+3, device=state.device)
			occupied = torch.isin(new_slots, state[1]).any() 

			use_source = torch.logical_and(is_source, torch.logical_not(occupied))
			new_values = torch.stack((
				torch.tensor([offset, self.op_token, token], device=state.device),
				new_slots
			))

			_, free_inds = torch.topk((state[1] < 0).int(), k=3, sorted=False)
			cur_values = state[:,free_inds]
			state[:,free_inds] = torch.where(use_source, new_values, cur_values)
			"""
			"""
			return state.clone(), token 

		init_state = torch.full((2,30), -1, device=self.fixed_offsets.device)
		# carry, out = scan_fn(init_state, None)
		_, tokens = scan(scan_fn, init_state, torch.arange(C, device=init_state.device))
		return tokens 


	def __getitem__old(self, index: int):
		gen = torch.Generator()
		gen.manual_seed(hash((self.seed, index)) % (2**32))
		# self.gen.manual_seed(self.seed + index)
		C = self.context_len
		V = self.num_vals + 1
		optoken = V - 1
		inds = torch.arange(C)

		# dest_mask has positions just after all CP tokens
		dest_mask = torch.randint(0, int(1 / self.op_frequency), (C,), generator=gen) == 0

		# avoids consecutive CP
		dest_mask[:-1] = torch.logical_xor(
			dest_mask[:-1], torch.logical_and(dest_mask[:-1], dest_mask[1:])
		)
		# avoids CP space CP
		dest_mask[:-2] = torch.logical_xor(
			dest_mask[:-2], torch.logical_and(dest_mask[:-2], dest_mask[2:])
		)
		dest_mask[:self.max_fixed_offset+1] = False # prevent OP too early
		dest_mask[C-2:] = False
		ops_mask = torch.full((C,), False)
		ops_mask[:-1] = dest_mask[1:] # positions of CP tokens 

		offset_mask = torch.full((C,), False)
		offset_mask[:-2] = dest_mask[2:] # positions of offset

		base = torch.randint(0, V-1, (C,), generator=gen)
		offset_vals = self.fixed_offsets[torch.randint(0, self.fixed_offsets.numel(), (C,))]
		base = torch.where(offset_mask, offset_vals, base)

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
