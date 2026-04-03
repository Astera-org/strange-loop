import equinox as eqx
import jax
from jax.experimental import checkify
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

B = begin-of-sequence token
S = start value
N = number of values to count
I = increment

B S N I D D D ... S N I D D D ...

This can test generalization over unseen (but still within the same vocabulary)
values for S, N or I, or any combination.

For example, with a vocabulary size of 50, S could in principle take on values [1,
48) or so.  This could be partitioned into disjoint subsets for train and test.


An example:

0 15  5  3 15 18 21 24 27 32 13  1 32 33 34 35 36 37 38 39 40 41 42 43 44  9  5  4  9
B  S  N  I D  D  D  D  D   S  N  I  D  D  D  D  D  D  D  D  D  D  D  D  D  S  N  I  D

13 17 21 25 47  2  1 47 48 23  3  9 23 32 41 42  3  1 42 43 44 46  3  1 46 47 48 39
 D  D  D  D  S  N  I  D  D  S  N  I  D  D  D  S  N  I  D  D  D  S  N  I  D  D  D  S

2  2 39 41 16  8  2 16 18 20 22 24 26 28 30 27  5  2 27 29 31 33 35 21  5  6 21 27 33
N  I  D  D  S  N  I  D  D  D  D  D  D  D  D  S  N  I  D  D  D  D  D  S  N  I  D  D  D

39 45 15 10  1 15 16 17 18 19 20 21 22 23
 D  D  S  N  I  D  D  D  D  D  D  D  D  D

"""
def _debug_assert(pred, data):
	I, maxi, S = data
	if not pred:
		raise ValueError(f"I must be positive.  Got {I=}, {maxi=}, {S=}")

def _Spos(pred, data):
	S, msg = data
	if not pred:
		raise ValueError(f"S must be positive.  {msg}: Got {S=}")


@dataclass
class StridedCountOpts:
	context_len: int
	vocab_size: int # includes BOS token
	geom_p: float   # p parameter for geoemtric pmf to sample N and S
	split_start: bool # whether to do train/test split on start positions
	split_incr: bool  # whether to do train/test split on incr positions
	start_train_frac: float # fraction of total start values to use for train
	incr_train_frac: float  # fraction of total incr values to use for train

	def __post_init__(self):
		assert 0 <= self.start_train_frac <= 1.0, (
			f"start_train_frac must be in [0, 1],  Got {self.start_train_frac}"
		)
		assert 0 <= self.incr_train_frac <= 1.0, (
			f"incr_train_frac must be in [0, 1].  Got {self.incr_train_frac}"
		)
		if self.split_start and (self.start_train_frac < 0.1 or self.start_train_frac > 0.9):
			raise RuntimeError(f"Bad split: {self.split_start=}, {self.start_train_frac=}")
		if self.split_incr and (self.incr_train_frac < 0.1 or self.incr_train_frac > 0.9):
			raise RuntimeError(f"Bad split: {self.split_incr=}, {self.incr_train_frac=}")


class StridedCountDataset(eqx.Module):
	opts: StridedCountOpts = eqx.field(static=True)
	start_pmf: jax.Array
	start_logpmf: jax.Array
	count_pmf: jax.Array
	count_logpmf: jax.Array
	start_val_mask: jax.Array
	incr_val_mask: jax.Array

	def __init__(self, opts: StridedCountOpts, is_train: bool, seed: int):
		self.opts = opts
		V = self.opts.vocab_size
		p = self.opts.geom_p
		key = jax.random.key(seed)
		k1, k2, k3 = jax.random.split(key, num=3)

		tmp1 = jnp.zeros(V)
		tmp1 = tmp1.at[-2:].set(-jnp.inf)
		tmp1 = tmp1.at[0].set(-jnp.inf)
		self.start_logpmf = tmp1
		self.start_pmf = jax.nn.softmax(tmp1)

		all_start = self.start_pmf > 0.0
		start = jfuncs.subsample_mask(k1, all_start, opts.start_train_frac)
		if opts.split_start and not is_train:
			start = jnp.logical_xor(all_start, start)
		self.start_val_mask = start

		all_incr = jnp.arange(V) > 0 
		incr = jfuncs.subsample_mask(k2, all_incr, opts.incr_train_frac)
		if opts.split_incr and not is_train:
			incr = jnp.logical_xor(all_incr, incr)
		self.incr_val_mask = incr

		tmp2 = jnp.pad(jnp.exp(jfuncs.geometric_logpmf(p, V - 1)), (1, 0))
		self.count_pmf = tmp2 # unnormalized - truncated to V values
		self.count_logpmf = jnp.log(tmp2)

		jax.debug.print("incr_val_mask: {}", self.incr_val_mask)

	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: Key) -> TokensAndProbs:
		V = self.opts.vocab_size
		true_val, false_val = jnp.array(True), jnp.array(False)

		def b0_start(key, _):
			log_dist = jnp.where(self.start_val_mask, self.start_logpmf, -jnp.inf)
			S = jax.random.categorical(key, log_dist)
			# jax.debug.callback(_Spos, S > 0, (S, "b0_start"))
			dist = jax.nn.softmax(log_dist)
			return jnp.array([1, S, -1, -1]), (S, dist, true_val, false_val)
	
		def b1_count(key, arg):
			S = arg[0]
			end = V - S + 1  
			# jax.debug.callback(_Spos, S > 0, (S, "b1_count"))
			log_mask = jfuncs.log_range_mask(2, end, V)
			logit_dist = self.count_logpmf + log_mask 
			N = jax.random.categorical(key, logit_dist)
			dist = jax.nn.softmax(logit_dist)
			# jax.debug.print("dist: {}", dist)
			return jnp.array([2, S, N, -1]), (N, dist, true_val, false_val)

		def b2_incr(key, arg):
			S, N = arg[:2]
			maxi = jax.lax.clamp(
				2, 
				jax.lax.floor((V - S) / (N - 1)).astype(jnp.int32), 
				V - 1
			)
			log_dist = jfuncs.log_range_mask(1, maxi, V)
			log_dist = jnp.where(self.incr_val_mask, log_dist, -jnp.inf)
			dist = jax.nn.softmax(log_dist)
			I = jax.random.categorical(key, log_dist)
			# jax.debug.callback(_debug_assert, I > 0, (I, maxi, S))

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
			state, arg = carry[0], carry[1:]
			return jax.lax.switch(state, branches, key, arg)

		def single_scan(key, length):
			init = jnp.array([4, -1, -1, -1])
			keys = jax.random.split(key, num=length)
			return jax.lax.scan(scan_fn, init, keys)

		batch_scan = jax.vmap(single_scan, in_axes=(0, None))
		_, content = batch_scan(key_B, self.opts.context_len)
		# _, content = single_scan(key_B[0], self.opts.context_len)
		# content = jax.tree.map(lambda x: x[None,:], content)
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
		if I == 0:
			print(f"I == 0")
			return False

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


