import equinox as eqx
import jax
import jax.numpy as jnp
from math import floor
from random import choice
from .types import TokensAndProbs

from jaxtyping import Int, Array, Key 
from dataclasses import dataclass

"""
A dataset exhibiting strided-counting, i.e. "count by 2 or count by 5"
The pattern overall will be:

S = start value
N = number of values to count
I = increment

S N I V V V ... S N I V V V ...
"""

def pseudo(ctx_len: int, vocab_size: int):
	carry = 0, -1, -1, -1 
	for _ in range(ctx_len):
		state, aux1, aux2, aux3 = carry
		match state:
			case 0:
				# no use of aux
				S = choice(range(vocab_size-2)) 
				yield S
				carry = 1, S, -1, -1
			case 1:
				S = aux1
				max_N = vocab_size - S - 1
				N = choice(range(1, max_N + 1))
				yield N
				carry = 2, S, N, -1
			case 2:
				S, N = aux1, aux2
				max_I = vocab_size - 1 if N == 1 else floor((vocab_size - S) / (N - 1))
				I = choice(range(1, max_I + 1))
				yield I
				carry = 3, S, I, S + I * (N - 1)
			case 3:
				V, I, L = aux1, aux2, aux3
				yield V 
				if V == L:
					carry = 0, -1, -1, -1
				else:
					carry = 3, V + I, I, L 



@dataclass
class StridedCountOpts:
	context_len: int
	vocab_size: int
	max_n: int


class StridedCount(eqx.Module):
	opts: StridedCountOpts = eqx.field(static=True)

	def __init__(self, opts: StridedCountOpts):
		self.opts = opts
	
	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:

		def branch0(key, _, _, _):
			S = jax.random.choice(key, self.opts.vocab_size - 2)
			return (1, S, -1, -1), S
	
		def branch1(key, S, _, _):
			maxn = self.opts.vocab_size - S - 1 
			N = jax.random.choice(key, maxn) + 1
			return (2, S, N, -1), N

		def branch2(key, S, N, _):
			maxi = jnp.where(
					N == 1, 
					self.opts.vocab_size - 1,
					floor((self.opts.vocab_size - S) / (N - 1))
					)
			I = jax.random.choice(maxi) + 1
			return (3, S, I, S + I * (N - 1)), I

		def branch3(key, V, I, L):
			carry = jax.lax.cond(
				V == L,
				lambda: 0, -1, -1, -1,
				lambda: 3, V + I, I, L
			)
			return carry, V

		branches = branch0, branch1, branch2, branch3

		# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
		def scan_fn(carry, key: Key[Array, ""]) -> tuple:
			"""
			"""
			state, *args = carry
			return jax.lax.switch(state, branches, key, *args)

		



