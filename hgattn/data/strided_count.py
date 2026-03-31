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

S N I D D D ... S N I D D D ...
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
		V = self.opts.vocab_size
		self.start_dist = jnp.ones(V) / (V - 2)
		self.start_dist[-2:] = 0.0
		self.log_start_dist = jnp.log(self.start_dist)
	
	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		V = self.opts.vocab_size
		true_val, false_val = jnp.array(True), jnp.array(False)

		def branch0(key, _):
			S = jax.random.categorical(key, self.log_start_dist)
			carry = jnp.array([1, S, -1, -1])
			content = S, self.start_dist, true_val, false_val
			return carry, content
	
		def branch1(key, arg):
			S = arg[0]
			maxn = V - S - 1 
			dist = jfuncs.range_mask(1, maxn, V).astype(jnp.float32)
			dist = dist / dist.sum()
			N = jax.random.choice(key, jnp.log(dist))
			carry = jnp.array([2, S, N, -1])
			content = N, dist, true_val, false_val 
			return carry, content

		def branch2(key, arg):
			S, N = arg[:2]
			maxi = jnp.where(N == 1, V - 1, floor((V - S) / (N - 1)))
			dist = jfuncs.range_mask(1, maxi, V).astype(jnp.float32)
			dist = dist / dist.sum()
			I = jax.random.choice(key, jnp.log(dist))
			carry = jnp.array([3, S, I, S + I * (N - 1)])
			content = I, dist, true_val, false_val 
			return carry, content

		def branch3(key, arg):
			D, I, L = arg
			carry = jnp.where(
				D == L, 
				jnp.array([0, -1, -1, -1])
				jnp.array([3, D + I, I, L])
			)
			content = D, jax.nn.one_hot(D, V), true_val, true_val 

		branches = branch0, branch1, branch2, branch3

		# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
		def scan_fn(carry: Int[Array], key: Key[Array, ""]) -> tuple:
			"""
			"""
			state, arg = carry[0], carry[1:]
			return jax.lax.switch(state, branches, key, arg)

		def singe_scan(key, length):
			init = jnp.array([0, -1, -1, -1])
			keys = jax.random.split(key, num=length)
			return jax.lax.scan(scan_fn, init, keys)

		batch_scan = jax.vmap(single_scan, in_axes=(0, None))
		_, content = batch_scan(key_B, self.opts.ctx_len)
		return TokensAndProbs(jax.random.key_data(key_B), *content)






		
		



