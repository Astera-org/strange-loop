from torch import Tensor
from torch.utils._pytree import register_pytree_node, tree_map
from dataclasses import dataclass

@dataclass
class TokensAndProbs:
	obs_sym: Tensor # int[context]
	obs_prob: Tensor # float[context, vocab]
	obs_mask: Tensor # bool[context]

	@staticmethod
	def _flatten(obj):
		return [obj.obs_sym, obj.obs_prob, obj.obs_mask], None

	@staticmethod
	def _unflatten(vals, ctx):
		return TokensAndProbs(*vals)

	def to(self, device):
		return tree_map(lambda x: x.to(device), self)


register_pytree_node(TokensAndProbs, 
					 TokensAndProbs._flatten,
					 TokensAndProbs._unflatten)

