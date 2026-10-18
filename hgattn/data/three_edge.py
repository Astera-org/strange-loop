from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import numpy as np
from ortools.sat.python import cp_model
from dataclasses import dataclass
from enum import StrEnum, auto
from . import DatasetOpts
from ..utils import MirroredEnum
from .. import jfuncs
from .types import TokensAndProbs

class Token(StrEnum):
	S = auto()
	P = auto()
	Q = auto()
	EQUALS = auto()
	EOS = auto()
	PAD = auto()

class Task(MirroredEnum):
	THREE_EDGE = "three-edge"   # [S-P Bag] [Q Q Q ...] = [S S S ...] EOS

@dataclass
class ThreeEdgeOpts(DatasetOpts):
	task: Task
	num_pq: int
	num_s: int
	num_nz: int
	x_rate: float
	train_frac: float

@dataclass
class TokenInfo:
	start: int
	vocab_size: int


class ThreeEdgeDataset(eqx.Module):
	opts: ThreeEdgeOpts = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	tokens: dict[Token, TokenInfo] = eqx.field(static=True)
	graph: Array

	def __init__(self, opts: ThreeEdgeOpts, is_train: bool, seed: int):
		self.opts = opts
		self.is_train = is_train
		self.seed = seed # currently unused

		key = jax.random.key(seed)
		I = opts.num_pq
		J = opts.num_pq
		K = opts.num_s

		self.tokens = {
				Token.S: TokenInfo(start=0, vocab_size=K),
				Token.P: TokenInfo(start=K, vocab_size=I),
				Token.Q: TokenInfo(start=K+I, vocab_size=J),
				Token.EQUALS: TokenInfo(start=K+I+J, vocab_size=1),
				Token.EOS: TokenInfo(start=K+I+J+1, vocab_size=1),
				Token.PAD: TokenInfo(start=K+I+J+2, vocab_size=1),
				}
		choice_fn = lambda key: jax.random.choice(key, K, (opts.num_nz,), replace=False)
		i, j, _ = jnp.ogrid[:I,:I,:1]
		inds = jax.vmap(choice_fn)(jax.random.split(key, num=I*I)).reshape(I, I, -1)
		self.graph = jnp.zeros((I, I, K), dtype=jnp.int32).at[i, j, inds].set(1)

	def _maybe_solve(self, key: PRNGKeyArray) -> tuple[Array, Array, Array]:
		"""
		Returns a representation of triplets (p, q, s) sorted by p in [0, I)
		q_I: the q value for triplet i
		s_I: the s value for triplet i 
		x_K: bool[K]: set of s present in the solution 
		feasible: bool[], whether the solution is feasible.  (if false, do not use)
		"""
		# return w, x, feasible
		I = self.opts.num_pq
		J = self.opts.num_pq
		K = self.opts.num_s
		xkey, wkey = jax.random.split(key)
		x_K = jax.random.bernoulli(xkey, self.opts.x_rate, K)
		marg_IJ = jnp.einsum('ijk, k -> ij', self.graph, x_K.astype(jnp.int32))
		marg_I = marg_IJ.sum(axis=1)

		# (i,j) choices with a unique k for the particular x_K
		valid_IJ = (marg_IJ == 1).astype(jnp.int32) 

		# i has at least one j which forms a valid (i,j) pair
		valid_I = valid_IJ.sum(axis=1)
		feasible = jnp.all(valid_I >= 1)
		cdf_IJ = jnp.cumsum(valid_IJ, axis=1) - 1
		target_I = jax.random.randint(wkey, shape=(I,), minval=0, maxval=valid_I)
		p_I = jnp.arange(I)
		q_I = jax.vmap(partial(jnp.searchsorted, side="left"))(cdf_IJ, target_I)
		chosen_IK = jnp.where(x_K[None,:], self.graph[p_I, q_I], 0)
		# chosen_IK, num_s = jfuncs.partition_masked(self.graph[p_I, q_I], x_K, axis=1)
		s_I = jnp.argmax(chosen_IK, 1) # 

		return q_I, s_I, x_K, feasible

	def _validate_solution(self, q_I, s_I, x_K) -> Array:
		I = self.opts.num_pq
		J = self.opts.num_pq
		K = self.opts.num_s
		s_valid = jnp.all(x_K[s_I])
		p_I = jnp.arange(I)
		chosen_IK = jnp.where(x_K[None,:], self.graph[p_I, q_I], 0)
		chosen_I = chosen_IK.sum(axis=1)
		qs_valid = jnp.all(chosen_I == 1)
		return s_valid & qs_valid 

	def _generate_three_edge(self, key: PRNGKeyArray) -> tuple[Array]:
		"""
		Generate the content of one item 
		obs_sym: [S-P Bag] [Q Q Q ...] = [S S S ...] EOS
		input_mask: bool array
		target_code: identifies r.h.s. tokens
		split_hash: hash for splitting into train / test
		"""
		I = self.opts.num_pq
		J = self.opts.num_pq
		K = self.opts.num_s
		L = K + 3 * I + 2
		keys = jax.random.split(key, num=3)
		q_I, s_I, x_K, feasible = self._maybe_solve(keys[0])
		p_I = jnp.arange(I)
		q_perm_I = jax.random.permutation(keys[1], I)
		s_K, num_s = jfuncs.partition_masked(jnp.arange(K), x_K)
		sizes = jnp.array([0, num_s, I, I, 1, I, 1])
		s_beg, p_beg, q_beg, eq_beg, t_beg, eos_beg, end = jnp.cumsum(sizes)

		equals_token = self.tokens[Token.EQUALS].start
		eos_token = self.tokens[Token.EOS].start
		pad_token = self.tokens[Token.PAD].start

		obs_sym = jnp.full(L, pad_token, dtype=jnp.int32)
		obs_sym = jfuncs.copy_range(obs_sym, s_K + self.tokens[Token.S].start, s_beg)
		obs_sym = jfuncs.copy_range(obs_sym, p_I + self.tokens[Token.P].start, p_beg)
		obs_sym = jfuncs.copy_range(
				obs_sym, q_I[q_perm_I] + self.tokens[Token.Q].start, q_beg)
		obs_sym = obs_sym.at[eq_beg].set(equals_token)
		obs_sym = jfuncs.copy_range(
				obs_sym, s_I[q_perm_I] + self.tokens[Token.S].start, t_beg)
		obs_sym = obs_sym.at[eos_beg].set(eos_token)
		obs_sym = jfuncs.shuffle_range(keys[2], obs_sym, 0, q_beg) 

		input_mask = jnp.arange(L) < eos_beg
		target_code = jnp.full(L, -1, dtype=jnp.int32)
		target_code = jfuncs.copy_range(target_code, jnp.arange(I), t_beg)

		split_hash = jfuncs.hash(obs_sym)
		return obs_sym, input_mask, target_code, split_hash, feasible

	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		gen_fn = self._generate_three_edge
		obs_sym, inp_mask, target_code, split_hash, feasible = gen_fn(key)
		is_train_frac = (split_hash % 1048576) < int(self.opts.train_frac * 1048576)
		is_active = (self.is_train == is_train_frac & feasible)

		return TokensAndProbs(
				key=jax.random.key_data(key),
				obs_sym=obs_sym,
				obs_prob=None,
				input_mask=inp_mask,
				target_code=target_code,
				active=is_active,
				)

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		item = eqx.filter_vmap(self._gen_one_item)(key_B)
		B = key_B.shape[0]
		train_size = int(B * self.opts.train_frac)
		size = train_size if self.is_train else B - train_size

		def _fraction(x):
			x, _ = jfuncs.partition_masked(x, item.active, 0)
			return x[:size]

		return jax.tree.map(_fraction, item)


if __name__ == "__main__":
	jnp.set_printoptions(linewidth=180)
	opts = ThreeEdgeOpts("three-edge", 10, 50, 2, 0.5, 0.75)
	ds = ThreeEdgeDataset(opts, True, 12345)
	key = jax.random.key(93848)
	q_I, s_I, x_K, feasible = ds._maybe_solve(key)
	is_valid = ds._validate_solution(q_I, s_I, x_K)

	key_B = jax.random.split(key, num=10)
	item = ds._gen_item(key_B)
	import pdb
	pdb.set_trace()

