from typing import Any
from torch import nn
from . import embed
from .embed import TokEmbedType, ValueMapEmbedding, PatchPositionEncoding

def make_token_embed(embed_ty: TokEmbedType, **kwargs) -> Any:
	match embed_ty:
		case TokEmbedType.STD:
			return nn.Embedding(**kwargs)
		case TokEmbedType.STD_POS:
			std = nn.Embedding(kwargs['num_embeddings'], kwargs['embedding_dim'])
			ptc = PatchPositionEncoding(kwargs['embedding_dim'], kwargs['ctx_len'])
			return nn.Sequential(std, ptc)
		case TokEmbedType.VALS_SHARED | TokEmbedType.VALS_SHARED_POS:
			n = kwargs['num_tokens']
			embed_dim = kwargs['embedding_dim']
			value_mult = [(0, float(i+1)) for i in range(n-1)] + [(n - 1, 1.0)]
			valmap_embed = ValueMapEmbedding(value_mult, n, embed_dim)
			if embed_ty == TokEmbedType.VALS_SHARED:
				return valmap_embed 
			else:
				ctx_len = kwargs['ctx_len']
				patch = PatchPositionEncoding(embed_dim, ctx_len)
				return nn.Sequential(valmap_embed, patch)

