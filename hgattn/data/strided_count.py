import equinox as eqx
import jax
import jax.numpy as jnp
from .types import TokensAndProbs

from jaxtyping import Int, Array, PRNGKeyArray
from dataclasses import dataclass

"""
A dataset exhibiting strided-counting, i.e. "count by 2 or count by 5"
The pattern overall will be:

S = start value
N = number of values to count
I = increment

S N I V V V ... S N I V V V ...
"""
@dataclass
class StridedCountOpts:
	context_len: int
	vocab_size: int
	max_n: int
	max_i: int


class StridedCount(eqx.Module):
	opts: StridedCountOpts = eqx.field(static=True)

	def __init__(self, opts: StridedCountOpts):
		self.opts = opts
	
	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		pass

