import numpy as np
import torch
from torch import nn
from pure_pytorch_reference import QuickGELU
from rotary_embedding_torch import RotaryEmbedding
from enum import Enum

class EmbedType(Enum):
	NONE = "none"
	GIVENS = "givens-rotation"
	ROPE = "rope"


class GraphAttention_Naive(nn.Module):
	def __init__(
			self, 
			d_model, 
			n_heads, 
			dropout_rate=0, 
			embed_type: EmbedType,
			head_subspaces: bool=False, 
			**kwargs
		):
		super(GraphAttention_Naive, self).__init__()

		# as with other small transformers, there are no head sub-spaces.
		# Really need to test if this is necessary!
		self.d_model = d_model
		self.n_heads = n_heads
		if head_subspaces:
			self.d_head = d_model//n_heads
		else:
			self.d_head = d_model
		self.head_subspaces = head_subspaces

		match embed_type:
			case EmbedType.NONE:
				self.embed = None
			case EmbedType.GIVENS:
				self.embed = 

		self.rotary_embed = None
		if use_rotary_embed:
			self.rotary_embed = RotaryEmbedding(self.d_head)

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wk = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)

		self.Wv = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)

		self.Wo = nn.Linear(d_model, d_model, bias=False, **kwargs)

		self._kscale = torch.tensor(np.sqrt(self.d_head) ** -1)
		self.register_buffer('kscale', self._kscale)

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

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

		if self.rotary_embed:
			Q = self.rotary_embed.rotate_queries_or_keys(Q)
			K = self.rotary_embed.rotate_queries_or_keys(K)

		A = torch.einsum('bhid,bhjd->bhij', Q, K)

		if target_mask is not None:
			if target_mask.ndim == 2:
				target_mask = target_mask[:,None,:]
			target_mask_BHQT = target_mask[:,None,:,:] # all heads masked the same
			A = torch.where(target_mask_BHQT, A, torch.tensor(float('-inf')))

		A = torch.softmax(A * self.kscale, dim=-1)
		y = torch.einsum('bhij,bhjd->bhid', A, V) # [batch_size, n_heads, ntok, d_head]

		# sum along the heads
		if self.head_subspaces:
			y = y.permute(0,2,1,3)
			y = y.reshape(batch_size, ntok, d_model)
		else:
			y = y.permute(0, 2, 3, 1).sum(dim=3).squeeze()
		# y = self.gelu(y)
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
