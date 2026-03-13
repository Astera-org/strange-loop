import jax
from jax import Array
import torch
from torch import Tensor
from jax.tree_util import register_pytree_node
from dataclasses import dataclass

@dataclass
class TokensAndProbs:
	obs_sym: Tensor|Array     # int[context]
	obs_prob: Tensor|Array    # float[context, vocab]
	input_mask: Tensor|Array  # bool[context]
	target_mask: Tensor|Array # bool[context]

	def to_torch(self):
		def convert(ten):
			ten.block_until_ready()
			return torch.utils.dlpack.from_dlpack(ten)
		return jax.tree.map(convert, self)


register_pytree_node(
	TokensAndProbs, 
	lambda x: ((x.obs_sym, x.obs_prob, x.input_mask, x.target_mask), None),
	lambda _, children: TokensAndProbs(*children)
)

