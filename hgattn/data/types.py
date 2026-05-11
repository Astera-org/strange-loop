import jax
from jax import Array
import torch
from torch import Tensor
from jax.tree_util import register_pytree_node
from dataclasses import dataclass

@dataclass
class TokensAndProbs:
	key: Tensor|Array         # random key
	obs_sym: Tensor|Array     # int[context]
	obs_prob: Tensor|Array    # float[context, vocab]
	input_mask: Tensor|Array  # bool[context]
	target_cat: Tensor|Array  # int[context], a category for each token, to partition
	                          # targets both for metrics and learning
	active: Tensor|Array      # bool, whether this item is active

	def to_torch(self):
		def convert(ten):
			ten.block_until_ready()
			return torch.utils.dlpack.from_dlpack(ten)
		return jax.tree.map(convert, self)

	def to_jax(self):
		def convert(ten):
			return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(ten))
		return jax.tree.map(convert, self)


register_pytree_node(
	TokensAndProbs, 
	lambda x: (
		(x.key, x.obs_sym, x.obs_prob, x.input_mask, x.target_cat, x.active), 
		None),
	lambda _, children: TokensAndProbs(*children)
)

