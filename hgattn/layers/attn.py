from dataclasses import dataclass
from typing import Any
from ..utils import MirroredEnum

class QKPosEncodingType(MirroredEnum):
	NONE = "none"
	GIVENS_RANDOM = "givens_random"
	GIVENS_ONE_HOT = "givens_one_hot"
	ROPE = "rope"

class AttnType(MirroredEnum):
	STD = "std"
	HYPERGRAPH = "hg"
	HYPERGRAPH_NAIVE = "hg_naive"
	UNIFORM = 'uni'

@dataclass
class AttentionOpts:
	qkv_bias: bool
	qk_norm: bool
	scatter: bool
	attn_ty: AttnType
	qk_pos_enc_ty: QKPosEncodingType
	qk_pos_enc_args: dict[str, Any]
