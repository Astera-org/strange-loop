from typing import Any
import torch
import torch.nn as nn
import math

from rotary_embedding_torch import RotaryEmbedding
from .givens_rotation import GivensRotation, InitType
from .attn import PosEmbedType
from einops import rearrange

class HypergraphAttentionNaive(nn.Module):
	"""Pure-PyTorch naive O(N^3) implementation for correctness testing."""
	def __init__(
			self, 
			d_model: int,
			n_heads: int, 
			d_head: int,
			pos_embed_type: PosEmbedType=PosEmbedType.NONE,
			pos_embed_args: dict[str, Any]=None,
			qrs_bias=False,
			scatter=False,
			dropout_rate=0,
			**kwargs
		):
		super().__init__()

		self.d_model = d_model
		self.n_heads = n_heads
		self.d_head = d_head
		self.pos_embed_type = pos_embed_type
		self.scatter = scatter

		match pos_embed_type:
			case PosEmbedType.NONE:
				self.embed = None
			case PosEmbedType.GIVENS_RANDOM:
				self.embed = GivensRotation(
					d_model, n_heads, d_head, init=InitType.RANDOM, **pos_embed_args)
			case PosEmbedType.GIVENS_ONE_HOT:
				self.embed = GivensRotation(
					d_model, n_heads, d_head, init=InitType.ONE_HOT, **pos_embed_args)
			case PosEmbedType.ROPE:
				self.embed = RotaryEmbedding(d_head, **pos_embed_args)

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)
		self.Wr = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)
		self.Ws = nn.Linear(d_model, self.d_head*n_heads, bias=qrs_bias, **kwargs)

		mult = 2 if self.scatter else 1
		self.Wv_q = nn.Linear(d_model, d_head*n_heads*mult, bias=qrs_bias, **kwargs)
		self.Wv_r = nn.Linear(d_model, d_head*n_heads*mult, bias=qrs_bias, **kwargs)
		self.Wv_s = nn.Linear(d_model, d_head*n_heads*mult, bias=qrs_bias, **kwargs)

		self.Wo = nn.Linear(self.d_model, d_model, bias=False, **kwargs)

		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x, mask=None):
		"""
		mask: bool[batch, query, target] if provided
		"""
		out_dtype = x.dtype
		x = x.float()
		B, C, M = x.shape
		H, D = self.n_heads, self.d_head

		Q = self.Wq(x)
		R = self.Wr(x)
		S = self.Ws(x)

		Q = Q.reshape(B, C, H, D).permute(0, 2, 1, 3)
		R = R.reshape(B, C, H, D).permute(0, 2, 1, 3)
		S = S.reshape(B, C, H, D).permute(0, 2, 1, 3)

		match self.pos_embed_type:
			case PosEmbedType.GIVENS_ONE_HOT | PosEmbedType.GIVENS_RANDOM:
				givens_mat = self.embed.compute_givens(x)
				Q = self.embed.rotate(givens_mat, Q)
				R = self.embed.rotate(givens_mat, R)
				S = self.embed.rotate(givens_mat, S)
			case PosEmbedType.ROPE:
				Q = self.embed.rotate_queries_or_keys(Q)
				R = self.embed.rotate_queries_or_keys(R)
				S = self.embed.rotate_queries_or_keys(S)
			case PosEmbedType.NONE:
				pass
			case _:
				raise RuntimeError(f"Unknown PosEmbedType: {self.pos_embed_type}")

		if self.scatter:
			# split the values into scatter and gather components
			Vq_full = self.Wv_q(x)
			Vr_full = self.Wv_r(x)
			Vs_full = self.Wv_s(x)
			Vq, Vq_ = Vq_full.reshape(B, C, H, D*2).permute(0, 2, 1, 3).split(D, dim=-1)
			Vr, Vr_ = Vr_full.reshape(B, C, H, D*2).permute(0, 2, 1, 3).split(D, dim=-1)
			Vs, Vs_ = Vs_full.reshape(B, C, H, D*2).permute(0, 2, 1, 3).split(D, dim=-1)
		else:
			Vq = self.Wv_q(x).reshape(B, C, H, D).permute(0, 2, 1, 3)
			Vr = self.Wv_r(x).reshape(B, C, H, D).permute(0, 2, 1, 3)
			Vs = self.Wv_s(x).reshape(B, C, H, D).permute(0, 2, 1, 3)

		dot_product = torch.einsum('bhid,bhjd,bhkd->bhijk', Q, R, S)
		dot_product = dot_product / (math.sqrt(D))

		dot_product_q = dot_product.flatten(3, 4) # BHI(JK)
		dot_product_r = dot_product.permute(0, 1, 3, 2, 4).flatten(3, 4) # BHJ(IK)
		dot_product_s = dot_product.permute(0, 1, 4, 2, 3).flatten(3, 4) # BHK(IJ)

		if mask is not None:
			if mask.ndim == 2:
				# mask is the standard 2D matrix (C, C)
				# add in a (dummy, broadcasted) batch dim
				mask = mask[None,:,:]
			# otherwise can have a different mask per batch element
			assert(mask.ndim == 3)
			# any i can attend to j,k <= i (likewise for the other 2 permutations).
			valid3 = (mask[:,:,:,None] & mask[:,:,None,:]).flatten(2, 3)
			invalid3 = ~(valid3[:,None,:,:])
			# the three dot_products are permuted and flattened so that the last dim is the softmax dim.
			dot_product_q = dot_product_q.masked_fill(invalid3, float('-inf'))
			dot_product_r = dot_product_r.masked_fill(invalid3, float('-inf'))
			dot_product_s = dot_product_s.masked_fill(invalid3, float('-inf'))

		Aq = torch.softmax(dot_product_q, dim=-1).reshape(dot_product.shape)
		Aq = torch.nan_to_num(Aq, nan=0.0)

		Ar = torch.softmax(dot_product_r, dim=-1).reshape(dot_product.shape)
		Ar = Ar.permute(0, 1, 3, 2, 4)
		Ar = torch.nan_to_num(Ar, nan=0.0)

		As = torch.softmax(dot_product_s, dim=-1).reshape(dot_product.shape)
		As = As.permute(0, 1, 3, 4, 2)
		As = torch.nan_to_num(As, nan=0.0)

		Y_q = torch.einsum('bhijk,bhjd,bhkd->bhid', Aq, Vr, Vs)
		Y_r = torch.einsum('bhijk,bhid,bhkd->bhjd', Ar, Vq, Vs)
		Y_s = torch.einsum('bhijk,bhid,bhjd->bhkd', As, Vq, Vr)
		# Y_q = self.gelu(Y_q)
		# Y_r = self.gelu(Y_r)
		# Y_s = self.gelu(Y_s)
		y = Y_q + Y_r + Y_s

		if self.scatter:
			# note: option for diamond op in scatter being 'add' removed.
			# (see README.md)
			Y_q_ = torch.einsum('bhijk,bhjd,bhijk,bhkd->bhid', Ar, Vr_, As, Vs_)
			Y_r_ = torch.einsum('bhijk,bhid,bhijk,bhkd->bhjd', Aq, Vq_, As, Vs_)
			Y_s_ = torch.einsum('bhijk,bhid,bhijk,bhjd->bhkd', Aq, Vq_, Ar, Vr_)
			# Y_q_ = self.gelu(Y_q_)
			# Y_r_ = self.gelu(Y_r_)
			# Y_s_ = self.gelu(Y_s_)
			y = y + Y_q_ + Y_r_ + Y_s_

		y = rearrange(y, 'b h i d -> b i (h d)')
		y = self.Wo(y)
		return y

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0.0
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model
		f += 3 * bs * ntok * d_model**2 * self.n_heads*d_model*2
		f += bs * self.n_heads * ntok**3 * d_model * 2
		f += bs * self.n_heads * ntok**3 * 2 * 3
		f += bs * self.n_heads * ntok**3 * d_model * 3
		f += bs * self.n_heads * ntok**3 * d_model * 3 * 3
		f += bs * self.n_heads * ntok * d_model * (6 + 6)
		f += bs * self.n_heads * ntok * d_model**2
		return f

