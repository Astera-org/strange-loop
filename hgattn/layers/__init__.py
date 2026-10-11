from typing import Any
from torch import nn
from .token_embed import (
	TokenEmbedOpts, StdTokenEmbedOpts, ValSharedTokenEmbedOpts, ValueMapEmbedding
)
from .input_pos_encoding import (
	InputPosEncodingOpts, NoEncodingOpts, LearnedEncodingOpts,
	SinusoidalPosEncoding, PatchPositionEncoding, SinusoidalPosEncoding
)

def make_input_layer(
	tok_embed: TokenEmbedOpts,
	input_pos_enc: InputPosEncodingOpts, 
) -> Any:
	match tok_embed:
		case StdTokenEmbedOpts():
			emb_module = nn.Embedding(tok_embed.num_tokens, tok_embed.embedding_dim)
		case ValSharedTokenEmbedOpts():
			n = tok_embed.num_tokens
			value_mult = [(0, float(i+1)) for i in range(n-1)] + [(n - 1, 1.0)]
			emb_module = ValueMapEmbedding(value_mult, n, tok_embed.embedding_dim)
		case _:
			raise RuntimeError(f"Unknown tok_embed opts: {tok_embed=}")

	match input_pos_enc:
		case NoEncodingOpts():
			pos_module = None
		case LearnedEncodingOpts():
			pos_module = PatchPositionEncoding(tok_embed.embedding_dim, input_pos_enc.ctx_len)
		case SinusoidalEncodingOpts():
			pos_module = SinusoidalPosEncoding(
				tok_embed.num_tokens, input_pos_enc.d_model, input_pos_enc.base,
				input_pos_enc.max_ctx
			)
		case _:
			raise RuntimeError(f"Unknown input_pos_enc: {input_pos_enc=}")

	if pos_module is None:
		return emb_module
	return nn.Sequential(emb_module, pos_module)
