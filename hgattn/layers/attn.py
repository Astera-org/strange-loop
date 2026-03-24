from dataclasses import dataclass
from typing import Any
from enum import Enum

class PosEmbedType(Enum):
	NONE = "none"
	GIVENS_RANDOM = "givens_random"
	GIVENS_ONE_HOT = "givens_one_hot"
	ROPE = "rope"


@dataclass
class AttentionOpts:
	impl: str
	qkv_bias: bool
	qk_norm: bool
	pos_ty: PosEmbedType
	pos_args: dict[str, Any]

	def __post_init__(self):
		try:
			self.pos_ty = PosEmbedType(self.pos_ty)
		except ValueError as v:
			raise ValueError(
					f"Received invalid pos_embed_type `{self.pos_ty.value}`.  "
					f"Valid ty's are {', '.join(m.value for m in PosEmbedType)}") from v

