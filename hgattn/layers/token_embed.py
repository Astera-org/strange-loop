from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from enum import Enum, StrEnum
import math

@dataclass
class TokenEmbedOpts:
	num_tokens: int
	embedding_dim: int

@dataclass
class StdTokenEmbedOpts(TokenEmbedOpts):
	ty: str = "std"

@dataclass
class ValSharedTokenEmbedOpts(TokenEmbedOpts):
	ty: str = "val_shared"


class ValueMapEmbedding(nn.Module):
	"""
	An embedding which maps a set of values to multiples of a given embedding.
	"""
	def __init__(
		self, 
		value_mult: list[tuple[int, float]],
		num_tokens: int,
		embedding_dim: int,
	):
		"""
		value_mult[i] = (embed_index, scale) associates a token i with an embedding
		vector and a scale.

		Final embedding vector assigned to token i is computed as:

		embed_index, scale = value_mult[i]
		embed = embedding[embed_index] * scale

		scale is fixed, but the embedding vectors are trainable.
		"""
		super().__init__()
		idx_mult = [value_mult[i] for i in range(num_tokens)]
		self._token_map = torch.tensor([p[0] for p in idx_mult])
		self._mult_map = torch.tensor([p[1] for p in idx_mult])

		self.register_buffer('mult_map', self._mult_map)
		self.register_buffer('token_map', self._token_map)

		num_embeddings = self.token_map.max() + 1
		self.raw_embed = nn.Embedding(num_embeddings, embedding_dim)
	
	def _forward(self, input_C: Tensor) -> Tensor:
		inds_C = self.token_map[input_C]
		mult_C = self.mult_map[input_C]
		embed_CV = self.raw_embed(inds_C)
		mult_embed_CV = embed_CV * mult_C[:,None] 
		return mult_embed_CV

	def forward(self, input_BC: Tensor) -> Tensor:
		return torch.vmap(self._forward)(input_BC)

