from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from enum import Enum


class TokEmbedType(Enum):
	STANDARD = "standard"
	FIRST_N_MULT = "first_n_mult"
	VALUE_MAP = "value_map"

@dataclass
class TokEmbedOpts:
	ty: TokEmbedType
	args: dict[str, Any]

	def __post_init__(self):
		try:
			self.ty = TokEmbedType(self.ty)
		except ValueError as v:
			raise ValueError(
					f"Received invalid tok_embed_type `{self.ty.value}`.  "
					f"Valid ty's are {', '.join(m.value for m in TokEmbedType)}") from v

class StandardEmbedding(nn.Module):
	"""
	Identical to nn.Embedding except with possible option of patching context
	position in the last channel
	"""
	def __init__(
		self,
		num_tokens: int,
		embedding_dim: int,
		splice_ctx_pos: bool,
	):
		super().__init__()
		self.splice_ctx_pos = splice_ctx_pos
		self.embed = nn.Embedding(num_tokens, embedding_dim)
		self.register_buffer('channel_mask_V', torch.full((embedding_dim,), False))
		self.channel_mask_V[-1] = True 
	
	def _forward(self, input_C: Tensor):
		embed_CV = self.embed(input_C)

		if self.splice_ctx_pos:
			C, = input_C.shape
			pos_C = torch.arange(C, dtype=input_C.dtype, device=input_C.device)
			embed_CV = torch.where(
					self.channel_mask_V[None,:], pos_C[:,None], embed_CV) 
		return embed_CV

	def forward(self, input_BC: Tensor) -> Tensor:
		return torch.vmap(self._forward)(input_BC)

class ValueMapEmbedding(nn.Module):
	"""
	An embedding which maps a set of values to multiples of a given embedding.
	"""
	def __init__(
		self, 
		value_mult: list[tuple[int, float]],
		num_tokens: int,
		embedding_dim: int,
		splice_ctx_pos: bool
	):
		"""
		value_mult[i] = (embed_index, scale) associates a token i with an embedding
		vector and a scale.

		Final embedding vector assigned to token i is computed as:

		embed_index, scale = value_mult[i]
		embed = embedding[embed_index] * scale

		scale is fixed, but the embedding vectors are trainable.

		if splice_ctx_pos, replace the last channel with ctx position
		"""
		super().__init__()
		idx_mult = [value_mult[i] for i in range(num_tokens)]
		self._token_map = torch.tensor([p[0] for p in idx_mult])
		self._mult_map = torch.tensor([p[1] for p in idx_mult])
		self.splice_ctx_pos = splice_ctx_pos

		self.register_buffer('mult_map', self._mult_map)
		self.register_buffer('token_map', self._token_map)
		self.register_buffer('channel_mask_V', torch.full((embedding_dim,), False))
		self.channel_mask_V[-1] = True 

		num_embeddings = self.token_map.max() + 1
		self.raw_embed = nn.Embedding(num_embeddings, embedding_dim)
	
	def _forward(self, input_C: Tensor) -> Tensor:
		inds_C = self.token_map[input_C]
		mult_C = self.mult_map[input_C]
		embed_CV = self.raw_embed(inds_C)
		mult_embed_CV = embed_CV * mult_C[:,None] 

		if self.splice_ctx_pos:
			C, = input_C.shape
			pos_C = torch.arange(C, dtype=mult_embed_CV.dtype, device=mult_embed_CV.device)
			mult_embed_CV = torch.where(
					self.channel_mask_V[None,:], pos_C[:,None], mult_embed_CV) 

		return mult_embed_CV

	def forward(self, input_BC: Tensor) -> Tensor:
		return torch.vmap(self._forward)(input_BC)

