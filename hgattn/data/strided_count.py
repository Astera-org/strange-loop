import equinox as eqx
import jax
import jax.numpy as jnp
from math import floor
from random import choice
from .types import TokensAndProbs
from .. import jfuncs

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


@dataclass
class StridedCountOpts:
	context_len: int
	vocab_size: int # includes BOS token
	geom_p: float   # p parameter for geoemtric pmf to sample N and S


class StridedCountDataset(eqx.Module):
	opts: StridedCountOpts = eqx.field(static=True)
	start_dist: jax.Array
	log_start_dist: jax.Array
	count_pmf: jax.Array
	count_logpmf: jax.Array

	def __init__(self, opts: StridedCountOpts):
		self.opts = opts
		V = self.opts.vocab_size
		p = self.opts.geom_p

		start_dist = jnp.ones(V)
		start_dist = start_dist.at[-2:].set(0.0)   # want non-empty runs
		start_dist = start_dist.at[0].set(0.0)     # BOS token
		tmp = jnp.concat((jnp.array([0]), jnp.exp(jfuncs.geometric_logpmf(p, V - 1))))
		self.count_pmf = tmp # unnormalized - truncated to V values
		self.count_logpmf = jnp.log(tmp)

		start_dist = start_dist / start_dist.sum()
		self.start_dist = start_dist
		self.log_start_dist = jnp.log(self.start_dist)
	
	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: Key) -> TokensAndProbs:
		V = self.opts.vocab_size
		true_val, false_val = jnp.array(True), jnp.array(False)

		def b0_start(key, _):
			S = jax.random.categorical(key, self.log_start_dist)
			carry = jnp.array([1, S, -1, -1])
			content = S, self.start_dist, true_val, false_val
			return carry, content
	
		def b1_count(key, arg):
			S = arg[0]
			end = V - S + 1  
			mask = jfuncs.log_range_mask(2, end, V).astype(jnp.float32)
			logit_dist = self.count_logpmf + mask 
			N = jax.random.categorical(key, logit_dist)
			dist = jax.nn.softmax(logit_dist)
			# jax.debug.print("dist: {}", dist)
			carry = jnp.array([2, S, N, -1])
			content = N, dist, true_val, false_val 
			return carry, content

		def b2_incr(key, arg):
			S, N = arg[:2]
			maxi = jax.lax.clamp(
				2, jax.lax.floor((V - S) / (N - 1)).astype(jnp.int32), V - 1
			)
			dist = jfuncs.range_mask(1, maxi, V).astype(jnp.float32)
			dist = dist / dist.sum()
			I = jax.random.categorical(key, jnp.log(dist))
			carry = jnp.array([3, S, I, S + I * (N - 1)])
			content = I, dist, true_val, false_val 
			return carry, content

		def b3_digit(key, arg):
			D, I, L = arg
			carry = jnp.where(
				D == L, 
				jnp.array([0, -1, -1, -1]),
				jnp.array([3, D + I, I, L])
			)
			content = D, jax.nn.one_hot(D, V), true_val, true_val 
			return carry, content

		def b4_bos(key, arg):
			carry = jnp.array([0, -1, -1, -1])
			content = 0, jax.nn.one_hot(0, V), true_val, false_val
			return carry, content

		branches = b0_start, b1_count, b2_incr, b3_digit, b4_bos

		# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
		def scan_fn(carry: Int[Array, "slots"], key: Key[Array, ""]) -> tuple:
			"""
			"""
			state, arg = carry[0], carry[1:]
			return jax.lax.switch(state, branches, key, arg)

		def single_scan(key, length):
			init = jnp.array([4, -1, -1, -1])
			keys = jax.random.split(key, num=length)
			return jax.lax.scan(scan_fn, init, keys)

		batch_scan = jax.vmap(single_scan, in_axes=(0, None))
		_, content = batch_scan(key_B, self.opts.context_len)
		return TokensAndProbs(jax.random.key_data(key_B), *content)


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

def validate_seq(sym: Array, vocab_size: int) -> bool:
	if sym[0] != 0:  # BOS token
		return False
	L = sym.shape[0]
	i = 0
	while i != L:
		i += 1
		if i == L:
			return True

		S = sym[i]
		if not 1 <= S < vocab_size - 2:
			print(f"S value out of range: {i}: {S=}")
			return False

		i += 1
		if i == L:
			return True

		N = sym[i]
		if not 1 <= N < vocab_size - S + 1:
			print(f"N value out of range: {i}: {N=} not in [1, {vocab_size-S+1})")
			return False
		
		i += 1
		if i == L:
			return True

		I = sym[i]
		last = S + I * (N - 1)
		if not last < vocab_size:
			print(f"Strided run exceeds maximal value: {i}: {last=}")
			return False

		target = S
		for _ in range(N):
			i += 1
			if i == L:
				return True
			if sym[i] != target:
				print(f"Incorrect strided value: {i}: {sym[i]=} != {target=}")
				return False
			target += I


