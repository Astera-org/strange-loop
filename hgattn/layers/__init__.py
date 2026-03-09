from typing import Any
from torch import nn
from . import embed

def make_token_embed(embed_type: embed.TokEmbedType, **kwargs) -> Any:
	match embed_type:
		case embed.TokEmbedType.STANDARD:
			return nn.Embedding(**kwargs)
		case embed.TokEmbedType.FIRST_N_MULT:
			n, r = kwargs['firstn'], kwargs['ntoks']
			embed_dim = kwargs['embed_dim']
			assert r >= n, "First N tokens cannot be greater than total number: {n} > {r}"
			value_mult = [(0, float(i+1)) for i in range(n)]
			value_mult.extend([(i + 1, 1.0) for i in range(r - n)])
			return embed.ValueMapEmbedding(value_mult, r, embed_dim)
		case embed.TokEmbedType.VALUE_MAP:
			return embed.ValueMapEmbedding(**kwargs)

