import equinox as eqx
from eqx.nn import Linear
import jax
import jax.numpy as jnp
from jaxtyping import Array
from dataclasses import dataclass
from .attn import PosEmbedType

@dataclass
class HypergraphAttentionOpts:
    d_model: int
    n_heads: int
    d_head: int
    qkv_bias: bool,
    qk_norm: bool,
    pos_embed_ty: PosEmbedType 


class HypergraphAttentionJ(eqx.Module):
    opts: HypergraphAttentionOpts 
    q_proj: Linear
    r_proj: Linear
    s_proj: Linear

    def __init__(self, opts):
        self.opts = opts

    def _one_forward(self, x, target_mask):
        """
        Compute self-attention on x
        Inputs:
        x: float32[context, d_model]
        target_mask: (one of)
           bool[target] - apply to all queries
           bool[query, target] - specific to each query
        Returns:
          float32[context, d_model]
        """
        nctx, d_model = x.shape


    def __call__(self, x, target_mask):
        return eqx.filter_jit(self._one_forward)(x, target_mask)



