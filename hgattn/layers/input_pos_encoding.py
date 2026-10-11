from dataclasses import dataclass
import torch
from torch import Tensor, nn

@dataclass
class InputPosEncodingOpts:
	pass

@dataclass
class NoEncodingOpts(InputPosEncodingOpts):
	ty: str = "none"

@dataclass
class LearnedEncodingOpts(InputPosEncodingOpts):
	ty: str = "learned"
	ctx_len: int = 0

@dataclass
class SinusoidalEncodingOpts(InputPosEncodingOpts):
	ty: str = "sinusoidal"
	num_tokens: int = 0
	d_model: int = 0
	base: int = 0
	max_ctx: int = 0


class LearnedPositionEncoding(nn.Module):
	"""
	Embedding transform that adds a learned positional encoding 
	"""
	def __init__(
		self,
		embedding_dim: int,
		ctx_len: int,
	):
		super().__init__()
		self.pos_encoding = nn.Parameter(torch.empty(1, ctx_len, embedding_dim))
		nn.init.trunc_normal_(self.pos_encoding, std=0.02)

	def forward(self, x_BCD: Tensor) -> Tensor:
		seq_len = x_BCD.shape[1]
		return x_BCD + self.pos_encoding[:, :seq_len, :]

class PatchPositionEncoding(nn.Module):
	"""
	An embedding transformation that simply patches in a (possibly scaled) context
	index position to the last channel.  The scale factor is learnable, and
	initialized to 1/ctx_length
	"""
	def __init__(
		self,
		embedding_dim: int,
		ctx_len: int,
	):
		super().__init__()
		self.scale = nn.Parameter(torch.full((), 1.0 / ctx_len))
		self.register_buffer('mask', torch.full((embedding_dim,), False))
		self.mask[-1] = True

	def _forward(self, input_CV: Tensor) -> Tensor:
		pos_C = torch.arange(
				input_CV.shape[0], dtype=input_CV.dtype, device=input_CV.device)
		return torch.where(self.mask[None,:], pos_C[:,None] * self.scale, input_CV)

	def forward(self, input_BCV: Tensor) -> Tensor: 
		return torch.vmap(self._forward)(input_BCV)


class SinusoidalPosEncoding(nn.Module):
	"""
	Produce the original Sinusoidal Positional Encoding from https://arxiv.org/pdf/1706.03762
	(section 3.5)
	"""
	def __init__(
		self,
		num_tokens: int,
		d_model: int,
		base: float=10000,
		max_ctx: int=100,
	):
		super().__init__()
		self.d_model = d_model
		theta_D = torch.pow(base, -torch.arange(0, d_model, 2) / d_model)
		self.register_buffer('theta_D', theta_D)

		pe_CD = torch.empty(max_ctx, self.d_model, dtype=theta_D.dtype) 
		pos_C = torch.arange(max_ctx, dtype=self.theta_D.dtype)
		pe_CD[:, 0::2] = torch.sin(pos_C[:,None] * self.theta_D[None,:])
		pe_CD[:, 1::2] = torch.cos(pos_C[:,None] * self.theta_D[None,:])
		self.register_buffer('pe_CD', pe_CD)

	def forward(self, embed_BCD: Tensor) -> Tensor:
		C = embed_BCD.shape[1]
		return embed_BCD * math.sqrt(self.d_model) + self.pe_CD[None,:C,:]

