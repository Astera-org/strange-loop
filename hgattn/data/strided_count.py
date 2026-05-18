import equinox as eqx
import jax
import numpy as np
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
	I, maxi, S, N, log_dist, log_dist_masked, key = data
	if not pred:
		raise ValueError(
			f"I must be positive.  Got {I=}, {maxi=}, {S=}, {N=} "
			f"{log_dist=} {log_dist_masked=}, {key=}")

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
	incr_min: jax.Array

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

		all_incr = jnp.arange(V) > 0 
		incr = jfuncs.subsample_mask(k2, all_incr, opts.incr_train_frac)
		if opts.split_incr and not is_train:
			incr = jnp.logical_xor(all_incr, incr)
		self.incr_val_mask = incr
		self.incr_min = jfuncs.first_index_of(self.incr_val_mask, True)
		start_end = V - self.incr_min

		all_start = jnp.logical_and(self.start_pmf > 0.0, jnp.arange(V) < start_end)
		start = jfuncs.subsample_mask(k1, all_start, opts.start_train_frac)
		if opts.split_start and not is_train:
			start = jnp.logical_xor(all_start, start)
		self.start_val_mask = start

		tmp2 = jnp.pad(jnp.exp(jfuncs.geometric_logpmf(p, V - 1)), (1, 0))
		self.count_pmf = tmp2 # unnormalized - truncated to V values
		self.count_logpmf = jnp.log(tmp2)

		# jax.debug.print("incr_val_mask: {}", self.incr_val_mask)

	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: Key) -> TokensAndProbs:
		V = self.opts.vocab_size
		true_val, false_val = jnp.array(True), jnp.array(False)

		def b0_start(key, state, arg):
			log_dist = jnp.where(self.start_val_mask, self.start_logpmf, -jnp.inf)
			S = jax.random.categorical(key, log_dist)
			# jax.debug.callback(_Spos, S > 0, (S, "b0_start"))
			dist = jax.nn.softmax(log_dist)
			return jnp.array([1, S, -1, -1]), (S, dist, true_val, false_val)
	
		def b1_count(key, state, arg):
			S = arg[0]
			end = (V - S) / self.incr_min + 1
			# jax.debug.callback(_Spos, S > 0, (S, "b1_count"))
			log_mask = jfuncs.log_range_mask(2, end, V)
			logit_dist = self.count_logpmf + log_mask 
			N = jax.random.categorical(key, logit_dist)
			dist = jax.nn.softmax(logit_dist)
			return jnp.array([2, S, N, -1]), (N, dist, true_val, false_val)

		def b2_incr(key, state, arg):
			S, N = arg[:2]
			end = jnp.where(N == 1, V, (V - S - 1e-5) / (N - 1))
			log_dist = jfuncs.log_range_mask(1, end, V)
			log_dist_masked = jnp.where(self.incr_val_mask, log_dist, -jnp.inf)
			dist = jax.nn.softmax(log_dist_masked)
			I = jax.random.categorical(key, log_dist_masked)
			good = jnp.logical_or(state != 2, I > 0)
			# jax.debug.callback(_debug_assert, good, (I, end, S, N, log_dist, log_dist_masked, key))

			carry = jnp.array([3, S, I, S + I * (N - 1)])
			content = I, dist, true_val, false_val 
			return carry, content

		def b3_digit(key, state, arg):
			D, I, L = arg
			carry = jnp.where(
				D == L, 
				jnp.array([0, -1, -1, -1]),
				jnp.array([3, D + I, I, L])
			)
			content = D, jax.nn.one_hot(D, V), true_val, true_val 
			return carry, content

		def b4_bos(key, state, arg):
			carry = jnp.array([0, -1, -1, -1])
			content = 0, jax.nn.one_hot(0, V), true_val, false_val
			return carry, content

		branches = b0_start, b1_count, b2_incr, b3_digit, b4_bos

		# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
		def scan_fn(carry: Int[Array, "slots"], key: Key[Array, ""]) -> tuple:
			state, arg = carry[0], carry[1:]
			return jax.lax.switch(state, branches, key, state, arg)

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

def validate_seq(ds: StridedCountDataset, sym: np.ndarray, stats: dict) -> bool:
	state = 4
	step = None
	it = iter(sym)
	for i, tok in enumerate(it):
		if not 0 <= tok < ds.vocab_size:
			print(f"{i}: Token out of bounds: {tok=}")
			return False
		match state:
			case 0: # start 
				if not ds.start_val_mask[tok]:
					print(f"{i}: S value not among legal values: S={tok}")
					return False
				start = tok
				state = 1
			case 1: # count
				if not 2 <= tok < ds.vocab_size:
					print(f"{i}: Count invalid: {tok=}")
					return False
				count = tok
				cts = stats.setdefault('count', {})
				cts.setdefault(count, 0)
				cts[count] += 1
				state = 2
			case 2: # incr
				if not ds.incr_val_mask[tok]:
					print(f"{i}: I value not among legal values: I={tok}")
					return False
				incr = tok
				state = 3
			case 3: # digit
				if step is None:
					step = 0
				digit = start + step * incr
				if tok != digit:
					print(f"{i}: expected {digit} but got token {tok}, {start=}, {step=}, {incr=}")
					return False
				step += 1
				if step == count:
					state = 0
					step = None
			case 4: # BOS
				if tok != 0:
					print(f"{i}: Expected BOS token 0, got {tok}")
					return False
				state = 0
	return True

