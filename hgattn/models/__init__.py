from .generative import GenerativeModel, GenerativeModelOpts
from ..layers.attn import AttentionOpts
from ..layers.token_embed import TokenEmbedOpts
from ..layers.input_pos_encoding import InputPosEncodingOpts
from ..debug import DebugOpts
from .base import ModelOpts
import torch

__all__ = ['make_model']

def make_model(
	arch: ModelOpts, 
	attn: AttentionOpts, 
	tok_embed: TokenEmbedOpts,
	input_pos_enc: InputPosEncodingOpts,
	debug: DebugOpts,
	seed: int
):
	match arch:
		case GenerativeModelOpts():
			return GenerativeModel(arch, attn, tok_embed, input_pos_enc, debug, seed)
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
