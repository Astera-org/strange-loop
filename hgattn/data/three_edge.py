from functools import partial
import math
import equinox as eqx
import numpy as np
from numpy.dtypes import StringDType
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array, PRNGKeyArray
import numpy as np
from dataclasses import dataclass
from enum import StrEnum, auto
from . import DatasetOpts
from ..utils import MirroredEnum
from .. import jfuncs
from .types import TokensAndProbs
from .base import BaseDataset
from ..tools.exact_cover import exact_cover, validate_cover

class Token(StrEnum):
	S = "S"
	P = "P" 
	Q = "Q" 
	EQUALS = "=" 
	EOS = "E" 
	PAD = "#" 

class Task(MirroredEnum):
	THREE_EDGE = "three-edge"   # [S-P Bag] [Q Q Q ...] = [S S S ...] EOS

@dataclass
class ThreeEdgeOpts(DatasetOpts):
	task: Task
	p: int
	q: int
	s: int
	nnz: int
	sfrac: float
	pfrac: float
	max_decoy: int
	train_frac: float

@dataclass
class TokenInfo:
	beg: int
	end: int


class ThreeEdgeDataset(BaseDataset):
	opts: ThreeEdgeOpts = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	tokens: dict[Token, TokenInfo] = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	graph_SPQ: Array

	def __init__(self, opts: ThreeEdgeOpts, is_train: bool, seed: int):
		super().__init__(is_train, opts.train_frac)
		self.opts = opts
		self.seed = seed # currently unused

		key = jax.random.key(seed)
		P = opts.p
		Q = opts.q
		S = opts.s

		self.tokens = {
				Token.S: TokenInfo(beg=0, end=S),
				Token.P: TokenInfo(beg=S, end=S+P),
				Token.Q: TokenInfo(beg=S+P, end=S+P+Q),
				Token.EQUALS: TokenInfo(beg=S+P+Q, end=S+P+Q+1),
				Token.EOS: TokenInfo(beg=S+P+Q+1, end=S+P+Q+2),
				Token.PAD: TokenInfo(beg=S+P+Q+2, end=S+P+Q+3),
				}
		choice_fn = lambda key: jax.random.choice(key, S, (opts.nnz,), replace=False)
		keys_PQ = jax.random.split(key, num=P*Q).reshape(P, Q)
		inds_2PQ = jax.vmap(jax.vmap(choice_fn, out_axes=1), out_axes=1)(keys_PQ)
		_, p, q = jnp.ogrid[:1,:P,:Q]

		self.graph_SPQ = jnp.zeros((S, P, Q), dtype=jnp.int32).at[inds_2PQ, p, q].set(1)
		self.vocab_size = self.tokens[Token.PAD].beg + 1

	def _generate_pool(self, key: PRNGKeyArray) -> tuple[Array]:
		"""
		Returns:
		mask_P: bool[P] mask defining presence of P tokens in the p-s pool 
		mask_S: bool[S] mask defining presence of S tokens in the p-s pool
		decoy_S: bool[S] mask for any extra S tokens that can be decoys
		found: bool[], whether the solution was found.  (if false, reject)
		"""
		# return w, x, feasible
		Q = self.opts.q
		P = self.opts.p
		S = self.opts.s
		pkey, wkey = jax.random.split(key)
		mask_P = jax.random.bernoulli(pkey, self.opts.pfrac, P)
		imask_P = mask_P.astype(jnp.int32)
		marg_SQ = (self.graph_SPQ * imask_P[None,:,None]).max(axis=1)
		state, found = exact_cover(marg_SQ)
		mask_S = state.chosen_R.astype(jnp.bool)
		decoy_S = (state.block_R == 0) & ~mask_S
		# is_valid = validate_cover(marg_SQ, mask_S | decoy_S) 

		# jax.debug.print("is_valid: {}", is_valid)

		# jax.debug.print("mask_S:  {}\ndecoy_S: {}\n", 
				  # mask_S.astype(jnp.int32), decoy_S.astype(jnp.int32))

		return mask_P, mask_S, decoy_S, found

	def _compute_target(self, mask_P, mask_S, perm_Q) -> Array:
		"""
		Computes the target values given the PS-pool and given Q permutation.
		Does not check uniqueness
		"""
		active_SPQ = self.graph_SPQ * mask_P[None,:,None] * mask_S[:,None,None]
		match_SQ = active_SPQ[:,:,perm_Q].max(axis=1)
		return jnp.argmax(match_SQ, axis=0)

	def _has_unique_target(self, mask_P, mask_S) -> Array:
		"""
		Return a bool[] denoting whether the target is unique
		"""
		active_SPQ = self.graph_SPQ * mask_P[None,:,None] * mask_S[:,None,None]
		active_SQ = active_SPQ.max(axis=1)
		return jnp.all(active_SQ.sum(axis=0) == 1)


	def _generate_one(self, key: PRNGKeyArray) -> TokensAndProbs:
		"""
		Generate the content of one item 
		obs_sym: [PS-pool] [Q Q Q ...] = [S S S ...] EOS
		input_mask: bool array
		target_code: identifies r.h.s. tokens
		split_hash: hash for splitting into train / test
		"""
		P = self.opts.p
		Q = self.opts.q
		S = self.opts.s

		avg_pool_size = P * self.opts.pfrac + S * self.opts.sfrac
		pool_budget = math.ceil(avg_pool_size * 1.5) + self.opts.max_decoy

		L = pool_budget + 2 * Q + 2
		keys = jax.random.split(key, num=3)
		mask_P, mask_S, decoy_S, feasible = self._generate_pool(keys[0])

		unique_target = self._has_unique_target(mask_P, mask_S | decoy_S)
		# jax.debug.print("unique_target: {}", unique_target)

		inds_Q = jax.random.permutation(keys[1], Q)
		inds_S, num_s = jfuncs.partition_masked(jnp.arange(S), mask_S)
		inds_P, num_p = jfuncs.partition_masked(jnp.arange(P), mask_P)
		decoy_S, num_decoy = jfuncs.partition_masked(jnp.arange(S), decoy_S)
		
		num_decoy = jnp.minimum(num_decoy, self.opts.max_decoy)
		decoy_S = jax.random.permutation(keys[2], decoy_S)

		in_budget = (num_s + num_p + num_decoy <= pool_budget)

		# compute the output
		target_Q = self._compute_target(mask_P, mask_S, inds_Q)

		sizes = jnp.array([0, num_s, num_p, num_decoy, Q, 1, Q, 1])
		s_beg, p_beg, d_beg, q_beg, eq_beg, t_beg, eos_beg, end = jnp.cumsum(sizes)

		equals_token = self.tokens[Token.EQUALS].beg
		eos_token = self.tokens[Token.EOS].beg
		pad_token = self.tokens[Token.PAD].beg
		s_token_beg = self.tokens[Token.S].beg
		q_token_beg = self.tokens[Token.Q].beg
		p_token_beg = self.tokens[Token.P].beg

		obs_sym = jnp.full(L, pad_token, dtype=jnp.int32)
		obs_sym = jfuncs.copy_range(obs_sym, inds_S + s_token_beg, s_beg, 0, num_s)
		obs_sym = jfuncs.copy_range(obs_sym, inds_P + p_token_beg, p_beg, 0, num_p)
		obs_sym = jfuncs.copy_range(obs_sym, decoy_S + s_token_beg, d_beg, 0, num_decoy)
		obs_sym = jfuncs.copy_range(obs_sym, inds_Q + q_token_beg, q_beg)
		obs_sym = obs_sym.at[eq_beg].set(equals_token)
		obs_sym = jfuncs.copy_range(obs_sym, target_Q + s_token_beg, t_beg)
		obs_sym = obs_sym.at[eos_beg].set(eos_token)
		obs_sym = jfuncs.shuffle_range(keys[3], obs_sym, 0, q_beg) 

		input_mask = jnp.arange(L) < eos_beg
		target_code = jnp.full(L, -1, dtype=jnp.int32)
		target_code = jfuncs.copy_range(target_code, jnp.arange(Q), t_beg)

		return TokensAndProbs(
			key=jax.random.key_data(key),
			obs_sym=obs_sym,
			obs_prob=None,
			input_mask=input_mask,
			target_code=target_code,
			active=(feasible & in_budget),
			split_hash=jfuncs.hash(obs_sym)
		)


	def validate(self, tokens: Array) -> Int[Array, ""]:
		tokens_ok = jnp.all((tokens >= 0) & (tokens < self.vocab_size))
		eq_mask = (tokens == self.tokens[Token.EQUALS].beg)
		eq_ok = eq_mask.sum() == 1
		eq_beg = jnp.argmax(eq_mask) 
		pos = jnp.arange(tokens.size)
		P = self.opts.p
		Q = self.opts.q
		S = self.opts.s

		def _vocab(code):
			info = self.tokens[code]
			return jnp.arange(info.beg, info.end)

		Q = self.opts.q
		T = Q # we're using each of the Q for the target
		
		q_mask = jnp.isin(tokens, _vocab(Token.Q))
		q_beg = jnp.argmax(q_mask)
		q_end = q_mask.size - jnp.argmax(q_mask[::-1])
		q_ok = (q_mask.sum() == Q) & ((q_end - q_beg) == Q)
		q_vals_T = tokens[q_beg + jnp.arange(Q)] - self.tokens[Token.Q].beg

		t_mask = jnp.isin(jnp.where(pos > eq_beg, tokens, -1), _vocab(Token.S))
		t_beg = jnp.argmax(t_mask)
		t_end = t_mask.size - jnp.argmax(t_mask[::-1])
		t_ok = (t_mask.sum() == Q) & ((t_end - t_beg) == Q)
		t_vals_T = tokens[t_beg + jnp.arange(Q)] - self.tokens[Token.S].beg

		p_active_P = jnp.isin(_vocab(Token.P), jnp.where(pos < eq_beg, tokens, -1))
		s_active_S = jnp.isin(_vocab(Token.S), jnp.where(pos < eq_beg, tokens, -1))

		pred_T = self._compute_target(p_active_P, s_active_S, q_vals_T)
		ans_correct_ok = jnp.all(t_vals_T == pred_T)
		ans_unique_ok = self._has_unique_target(p_active_P, s_active_S)

		status = jnp.array(0, dtype=jnp.int32)
		status = jnp.where(tokens_ok, status, 6)
		status = jnp.where(eq_ok, status, 5)
		status = jnp.where(q_ok, status, 4)
		status = jnp.where(t_ok, status, 3)
		status = jnp.where(ans_correct_ok, status, 2)
		status = jnp.where(ans_unique_ok, status, 1)

		# jax.debug.breakpoint()

		return status

	def print_raw_item(self, item: TokensAndProbs) -> str:
		B, L = item.obs_sym.shape

		def is_between(tokens, beg, end):
			return (tokens >= beg) & (tokens < end)

		codes = np.array([str(c) for c in Token])

		begs = jnp.array([self.tokens[c].beg for c in Token])
		ends = jnp.array([self.tokens[c].end for c in Token])
		mask = jax.vmap(is_between, in_axes=(None,0,0))(item.obs_sym, begs, ends)
		tags = jnp.argmax(mask, axis=0)
		vals = item.obs_sym - begs[tags]

		max_dim = max(self.opts.p, self.opts.q, self.opts.s)

		w = len(str(max_dim))
		out = np.empty((B, 2, L), dtype=f"<U{w}")
		out[:,0,:] = np.array(vals).astype(np.str_)
		out[:,1,:] = codes[tags]

		out = out[item.active] 

		result = "\n\n".join(
				"\n".join(
					" ".join(f"{el:>{w}}" for el in row) 
					for row in pair) 
				for pair in out)
		return result



if __name__ == "__main__":
	jnp.set_printoptions(linewidth=205, )
	opts = ThreeEdgeOpts(
		task="three-edge", p=5, q=5, s=100, nnz=2, sfrac=0.10, pfrac=0.8, train_frac=0.75)
	ds = ThreeEdgeDataset(opts, True, 12345)
	key = jax.random.key(93848)
	key_B = jax.random.split(key, num=100)
	mask_P, mask_S, feasible = jax.vmap(ds._generate_pool)(key_B)

	key_B = jax.random.split(key, num=10)
	item = ds._gen_batch(key_B)
	status = ds.validate_item(item)
	result = ds.print_raw_item(item)
	import pdb
	pdb.set_trace()


