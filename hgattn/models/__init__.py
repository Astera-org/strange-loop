from typing import Any
from .generative import GenerativeModel, GenerativeModelOpts
from ..layers.attn import AttentionOpts
from ..layers.embed import TokEmbedOpts  


__all__ = ['make_model']

def make_model(
	arch: Any, 
	attn: AttentionOpts, 
	embed: TokEmbedOpts, 
	seed: int
):
	match arch:
		case GenerativeModelOpts():
			return GenerativeModel(arch, attn, embed, seed)
		case default:
			raise NotImplementedError

