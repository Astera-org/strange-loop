from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn
from enum import Enum

class PosEmbedType(Enum):
	NONE = "none"
	GIVENS = "givens"
	ROPE = "rope"


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

@dataclass
class PosEmbedOpts:
	ty: PosEmbedType
	args: dict[str, Any]

	def __post_init__(self):
		try:
			self.ty = PosEmbedType(self.ty)
		except ValueError as v:
			raise ValueError(
					f"Received invalid pos_embed_type `{self.ty.value}`.  "
					f"Valid ty's are {', '.join(m.value for m in PosEmbedType)}") from v



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
	
	def forward(self, input_BC: Tensor) -> Tensor:
		"""
		"""
		inds_BC = self.token_map[input_BC]
		mult_BC = self.mult_map[input_BC]
		embed_BCV = self.raw_embed(inds_BC)
		mult_embed = embed_BCV * mult_BC[:,:,None] 
		return mult_embed

