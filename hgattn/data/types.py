from torch import Tensor
from dataclasses import dataclass

@dataclass
class TokensAndProbs:
	obs_sym: Tensor # int[batch, context]
	obs_prob: Tensor # float[batch, context, vocab]
	obs_mask: Tensor # bool[batch, context]
