from typing import Any
import numpy as np
import torch
from torch import nn
# from pure_pytorch_reference import QuickGELU
from rotary_embedding_torch import RotaryEmbedding
from .givens_rotation import GivensRotation
from .embed import PosEmbedType
from einops import rearrange


class GraphAttention_Naive(nn.Module):
	def __init__(
			self, 
			d_model: int,
			n_heads: int, 
			d_head: int,
			dropout_rate=0, 
			pos_embed_type: PosEmbedType=PosEmbedType.NONE,
			pos_embed_args: dict[str, Any]=None,
			**kwargs
		):
		super(GraphAttention_Naive, self).__init__()

		# as with other small transformers, there are no head sub-spaces.
		# Really need to test if this is necessary!
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_head = d_head
		self.pos_embed_type = pos_embed_type

		match pos_embed_type:
			case PosEmbedType.NONE:
				self.embed = None
			case PosEmbedType.GIVENS:
				self.embed = GivensRotation(d_model, n_heads, d_head, **pos_embed_args)
			case PosEmbedType.ROPE:
				self.embed = RotaryEmbedding(d_head, **pos_embed_args)

		qkv_bias = True
		# qkv_bias = False

		self.Wq = nn.Linear(d_model, d_head*n_heads, bias=qkv_bias, **kwargs)
		self.Wk = nn.Linear(d_model, d_head*n_heads, bias=qkv_bias, **kwargs)
		self.Wv = nn.Linear(d_model, d_head*n_heads, bias=qkv_bias, **kwargs)
		self.Wo = nn.Linear(d_head*n_heads, d_model, bias=False, **kwargs)

		self._kscale = torch.tensor(np.sqrt(d_head) ** -1)
		self.register_buffer('kscale', self._kscale)

		self.dropout = nn.Dropout(dropout_rate)
		# self.gelu = QuickGELU()

	def forward(self, x, target_mask):
		"""
		Compute self-attention on x
		Inputs:
		x: [batch, context, model]
		target_mask: one of:
			bool[batch, target] - a target mask applied to all queries
			bool[batch, query, target] - a target mask specific to each query
		"""
		batch_size, ntok, d_model = x.shape

		Q = self.Wq(x) # [batch, ctx, n_heads*d_head]
		K = self.Wk(x)
		V = self.Wv(x)

		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head)
		K = K.reshape(batch_size, ntok, self.n_heads, self.d_head)
		V = V.reshape(batch_size, ntok, self.n_heads, self.d_head)

		Q = Q.permute(0, 2, 1, 3) # [batch_size, n_heads, ntok, d_head]
		K = K.permute(0, 2, 1, 3)
		V = V.permute(0, 2, 1, 3)

		match self.pos_embed_type:
			case PosEmbedType.GIVENS:
				givens_mat = self.embed.compute_givens(x)
				Q = self.embed.rotate(givens_mat, Q)
				K = self.embed.rotate(givens_mat, K)
			case PosEmbedType.ROPE:
				Q = self.embed.rotate_queries_or_keys(Q)
				K = self.embed.rotate_queries_or_keys(K)
			case PosEmbedType.NONE:
				pass
			case default:
				raise RuntimeError(f"Unknown PosEmbedType: {self.pos_embed_type}")

		A = torch.einsum('bhid,bhjd->bhij', Q, K)

		if target_mask is not None:
			if target_mask.ndim == 2:
				target_mask = target_mask[:,None,:]
			target_mask_BHQT = target_mask[:,None,:,:] # all heads masked the same
			A = torch.where(target_mask_BHQT, A, torch.tensor(float('-inf')))

		A = torch.softmax(A * self.kscale, dim=-1)
		y = torch.einsum('bhij,bhjd->bhid', A, V) # [batch_size, n_heads, ntok, d_head]

		y = rearrange(y, 'b h i d -> b i (h d)')
		y = self.Wo(y)
		# residual path is external to this layer.
		return y

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		# QKV projection
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		# attention
		f += bs * self.n_heads * ntok**2 * d_model
		# softmax
		f += bs * self.n_heads * ntok**2 * 2
		# V projection
		f += bs * self.n_heads * ntok * d_model**2 * 2
		# sum and gelu
		f += bs * self.n_heads * ntok * d_model * (2 + 6)
		# Wo proj
		f += bs * ntok * d_model**2
		return f
