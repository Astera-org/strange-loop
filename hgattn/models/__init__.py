from typing import Any
from .generative import GenerativeModel, GenerativeModelOpts
from ..layers.attn import AttentionOpts
from ..layers.embed import TokEmbedOpts  
from ..debug import DebugOpts


__all__ = ['make_model']

def make_model(
	arch: Any, 
	attn: AttentionOpts, 
	embed: TokEmbedOpts, 
	debug: DebugOpts,
	seed: int
):
	match arch:
		case GenerativeModelOpts():
			return GenerativeModel(arch, attn, embed, debug, seed)
		case default:
			raise NotImplementedError

