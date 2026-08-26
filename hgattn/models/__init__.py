from typing import Any
from .generative import GenerativeModel, GenerativeModelOpts
from ..layers.attn import AttentionOpts
from ..layers.embed import TokEmbedOpts  
from ..debug import DebugOpts
import torch


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
		case _:
			raise NotImplementedError

@torch.no_grad()
def scale_model_weights(model: torch.nn.Module, factor: float) -> None:
	"""
	Scale all model weights by factor, in place
	"""
	for p in model.parameters():
		p.mul_(factor)

@torch.no_grad()
def weight_norm(model: torch.nn.Module) -> torch.Tensor:
	sq = sum(p.pow(2).sum() for p in model.parameters())
	return torch.as_tensor(sq).sqrt()
