from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from enum import Enum


class TokEmbedType(Enum):
	STD = "std"
	STD_POS = "std_pos"
	VALS_SHARED = "vals_shared"
	VALS_SHARED_POS = "vals_shared_pos"

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


