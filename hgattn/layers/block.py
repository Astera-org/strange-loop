import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from enum import Enum
from . import ffn
from .graph_attn import GraphAttention_Naive
from .attn import PosEmbedType

class FFNType(Enum):
	SWIGLU = "swiglu"
	MLP = "mlp"

class NormType(Enum):
	RMS_NORM = "rmsnorm"
	LAYER_NORM = "layernorm"

class TransformerBlock(nn.Module):
	def __init__(
		self,
		model_dim: int,
		num_heads: int,
		d_head: int,
		qkv_bias: bool,
		pos_ty: PosEmbedType,
		pos_args: dict[str, Any],
		hidden_dim: int,
		ffn_type: FFNType,
		norm_type: NormType,
		use_norm1: bool,
		use_norm2: bool,
	):
		super().__init__()
		match norm_type:
			case NormType.RMS_NORM:
				self.norm1 = nn.RMSNorm(model_dim) if use_norm1 else nn.Identity
				self.norm2 = nn.RMSNorm(model_dim) if use_norm2 else nn.Identity 
			case NormType.LAYER_NORM:
				self.norm1 = nn.LayerNorm(model_dim) if use_norm1 else nn.Identity 
				self.norm2 = nn.LayerNorm(model_dim) if use_norm2 else nn.Identity 
			case default:
				raise RuntimeError(f"Unrecognized NormType: {norm_type}")

		match ffn_type:
			case FFNType.SWIGLU:
				self.ffn = ffn.SwiGLU(model_dim, hidden_dim, model_dim)
			case FNNType.MLP:
				self.ffn = ffn.MLP(model_dim, hidden_dim, model_dim)

		self.attn = GraphAttention_Naive(
			model_dim, 
			num_heads, 
			d_head,
			qkv_bias,
			pos_embed_type=pos_ty,
			pos_embed_args=pos_args,
		)

	def forward(self, x, mask):
		xn1 = self.norm1(x)
		att = self.attn(xn1, mask)
		x = x + att
		xn2 = self.norm2(x)
		ffn = self.ffn(xn2)
		return x + ffn

