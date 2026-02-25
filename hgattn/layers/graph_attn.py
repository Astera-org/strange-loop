import torch
from torch import nn
from pure_pytorch_reference import QuickGELU


class GraphAttention_Naive(nn.Module):
	def __init__(self, d_model, n_heads, dropout_rate=0, head_subspaces=False, **kwargs):
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

		self.Wq = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)
		self.Wk = nn.Linear(d_model, self.d_head*n_heads, bias=False, **kwargs)

		self.Wv = nn.Linear(d_model, self.d_head*n_heads, bias=True, **kwargs)

		self.Wo = nn.Linear(d_model, d_model, bias=True, **kwargs)

		self.dropout = nn.Dropout(dropout_rate)
		self.gelu = QuickGELU()

	def forward(self, x, rotary_emb, target_mask):
		"""
		Compute self-attention on x
		Inputs:
		x: [batch, context, model]
		target_mask: one of:
			bool[batch, target] - a target mask applied to all queries
			bool[batch, query, target] - a target mask specific to each query
		"""
		batch_size, ntok, d_model = x.shape

		if rotary_emb is not None:
			Q = rotary_emb.rotate_queries_or_keys(self.Wq(x))
			K = rotary_emb.rotate_queries_or_keys(self.Wk(x))
		else:
			Q = self.Wq(x)
			K = self.Wk(x)
		V = self.Wv(x)

		Q = Q.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		K = K.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		# Q,K are hence [batch_size, n_heads, ntok, d_head]

		V = V.reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
		# V is [batch_size, n_heads, ntok, d_head]

		A = torch.einsum('bhid,bhjd->bhij', Q, K)

		if target_mask is not None:
			if target_mask.ndim == 2:
				target_mask = target_mask[:,None,:]
			target_mask_BHQT = target_mask[:,None,:,:] # all heads masked the same
			A = torch.where(target_mask_BHQT, A, torch.tensor(float('-inf')))

		A = torch.softmax(A, dim=-1)
		y = torch.einsum('bhij,bhjd->bhid', A, V)

		# sum along the heads
		if self.head_subspaces:
			y = y.permute(0,2,1,3)
			y = y.reshape(batch_size, ntok, d_model)
		else:
			y = y.permute(0, 2, 3, 1).sum(dim=3).squeeze()
		y = self.gelu(y)
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
