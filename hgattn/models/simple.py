import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pure_pytorch_reference import (
		HypergraphAttention_Naive, QuickGELU
		)
from ..layers.graph_attn import GraphAttention_Naive

from hypergraph_attention import HypergraphAttentionCPP

@dataclass
class SimpleCompOpts:
	num_tokens: int
	model_dim: int
	mlp_hidden_dim: int
	num_heads: int
	n_layers: int
	attn_impl: str
	n_recurse: int


class SwiGLU(nn.Module):
	"""
	Swish Gated Linear Units based Feed-Forward Network.
	"""
	def __init__(self, in_features, hidden_features, out_features):
		super().__init__()
		self.w1 = nn.Linear(in_features, hidden_features)
		self.w2 = nn.Linear(in_features, hidden_features)
		self.w3 = nn.Linear(hidden_features, out_features)

	def forward(self, x):
		return self.w3(F.silu(self.w1(x)) * self.w2(x))

class SimpleCompModel(nn.Module):
	"""Model with hypergraph attention layer."""
	def __init__(
			self, 
			num_tokens: int, model_dim: int, mlp_hidden_dim: int,
			num_heads: int, n_layers: int, attn_impl: str='', 
			n_recurse: int=1
			): 
		super().__init__()
		self.num_tokens = num_tokens
		self.embedding_proj = nn.Embedding(num_tokens, model_dim)
		# self.rotary_emb = RotaryEmbedding(dim = model_dim)
		self.attn_impl = attn_impl
		self.n_recurse = n_recurse
		self.d_model = model_dim

		def ignore_second(fun):
			def call(a, b):
				return fun(a)
			return call

		self.repeated_layers = nn.ModuleList()
		for _ in range(n_layers):
			if attn_impl == "hypergraph-naive":
				attention_layer = HypergraphAttention_Naive(model_dim, num_heads, head_subspaces=True)
			elif attn_impl == "hypergraph-tiled":
				hyper_attn = HypergraphAttentionCPP(model_dim, num_heads, dropout_rate=0)
				attention_layer = ignore_second(hyper_attn)
			elif attn_impl == "graph":
				attention_layer = GraphAttention_Naive(model_dim, num_heads, head_subspaces=True)
			else:
				raise RuntimeError(
						f"attn_impl must be one of 'hypergraph-naive', 'hypergraph-tiled', or 'graph'")

			norm1_layer = nn.RMSNorm(model_dim) # was LayerNorm
			norm2_layer = nn.RMSNorm(model_dim)
			# norm1_layer = nn.LayerNorm(model_dim)
			# norm2_layer = nn.LayerNorm(model_dim)
			if True:
				ffn_layer = nn.Sequential(
						nn.Linear(model_dim, mlp_hidden_dim),
						nn.ReLU(),
						nn.Linear(mlp_hidden_dim, model_dim)
						)
			else:
				ffn_layer = SwiGLU(model_dim, mlp_hidden_dim, model_dim)
				# keep the same number of parameters.

			self.repeated_layers.append(
					nn.ModuleDict({
						'attention': attention_layer,
						'norm1': norm1_layer,
						'ffn': ffn_layer,
						'norm2': norm2_layer,
						})
					)
		# self.output_proj = nn.Linear(model_dim, self.num_tokens)
		self.gelu = QuickGELU()

	def forward(self, x, mask):
		# skip = b % (self.n_recurse)
		skip = 0
		if skip > 0:
			with torch.no_grad():
				x = self.embedding_proj(x)
		else:
			x = self.embedding_proj(x)
		for r in range(self.n_recurse):
			if r < skip:
				with torch.no_grad():
					for layer_block in self.repeated_layers:
						# attn_output = layer_block['attention'](x, self.rotary_emb)
						xn = layer_block['norm1'](x) # PreNorm
						attn_output = layer_block['attention'](xn, None, mask)
						x = x + attn_output
						xn = layer_block['norm2'](x)
						ffn_output = layer_block['ffn'](xn)
						x = x + ffn_output
			else:
				for layer_block in self.repeated_layers:
					# attn_output = layer_block['attention'](x, self.rotary_emb)
					xn = layer_block['norm1'](x)
					attn_output = layer_block['attention'](xn, None, mask)
					x = x + attn_output
					xn = layer_block['norm2'](x)
					ffn_output = layer_block['ffn'](xn)
					x = x + ffn_output
					# attn_output = layer_block['attention'](x, None)
					# x = layer_block['norm1'](x + attn_output)
					# ffn_output = layer_block['ffn'](x)
					# x = layer_block['norm2'](x + ffn_output)

		return x
	# return self.output_proj(x)

	def save_model(self, path: str):
		"""Saves the model's configuration and state dictionary."""
		torch.save(self.state_dict(), path)
		print(f"saved model to {path}")

	def load_model(self, path: str, device):
		"""Loads a model from a file."""
		checkpoint = torch.load(path, map_location=device)
		self.load_state_dict(checkpoint)
		self.to(device)
		self.eval() # Set to evaluation mode by default
		return

	def printParamCount(self):
		trainable_params = sum(
				p.numel() for p in self.parameters() if p.requires_grad
				)
		print(f"SimpleCompModel {self.attn_impl}: number of model parameters:{trainable_params/1e6}M")

	def calcFlops(self, x):
		bs, ntok, d_model = x.shape
		f = 0
		for r in range(self.n_recurse):
			for layer_block in self.repeated_layers:
				f += layer_block['attention'].calcFlops(x)
				f += bs * ntok * self.d_model * 10 # layerNorm 1
				f += bs * ntok * self.d_model**2 * 3 * 2 # ffn
				f += bs * ntok * self.d_model * 10 # layerNorm 2
		f += bs * ntok * self.d_model**2 * self.input_dim # output proj
		return f

	def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
		# this is from nanoGPT!
		# start with all of the candidate parameters
		param_dict = {pn: p for pn, p in self.named_parameters()}
		# filter out those that do not require grad
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
		# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
		# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [
				{'params': decay_params, 'weight_decay': weight_decay},
				{'params': nodecay_params, 'weight_decay': 0.0}
				]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
		print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
		# Create AdamW optimizer and use the fused version if it is available
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == 'cuda'
		extra_args = dict(fused=True) if use_fused else dict()
		extra_args = {**extra_args, 'amsgrad': False}
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
		# optimizer = torch.optim.NAdam(optim_groups, lr=learning_rate)
		print(f"using fused AdamW: {use_fused}")

		return optimizer

