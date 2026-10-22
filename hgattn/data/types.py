import equinox as eqx
import jax
from jax import Array
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass

class TokensAndProbs(eqx.Module):
	key: Tensor|Array           # random key
	obs_sym: Tensor|Array       # int[context]
	obs_prob: Tensor|Array|None # float[context, vocab]
	input_mask: Tensor|Array    # bool[context], which input tokens are attended to
	target_code: Tensor|Array   # int[context], a category for each token, to partition
	                            # targets both for metrics and learning.
							    # interpretation is application dependent
	active: Tensor|Array        # bool, whether this item is active
	split_hash: Tensor|Array    # guarantees disjoint train / test split

	def to_torch(self):
		def convert(ten):
			ten.block_until_ready()
			return torch.utils.dlpack.from_dlpack(ten)
		return jax.tree.map(convert, self)

	def to_jax(self):
		def convert(ten):
			return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(ten))
		return jax.tree.map(convert, self)

	def to_numpy(self):
		def convert(ten):
			match ten:
				case Tensor():
					return ten.detach().cpu().numpy()
				case Array():
					return np.asarray(ten)
				case np.ndarray():
					return ten
				case _:
					return ten
		return jax.tree.map(convert, self)
