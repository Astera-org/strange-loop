from functools import partial
import math
import equinox as eqx
import numpy as np
from numpy.dtypes import StringDType
import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Array, PRNGKeyArray
from typing import Any
import numpy as np
from dataclasses import dataclass
from enum import StrEnum, auto
from . import DatasetOpts
from ..utils import MirroredEnum
from .. import jfuncs
from ..jfuncs import copy_range, shuffle_range
from .types import TokensAndProbs
from .base import BaseDataset
from ..tools.exact_cover import exact_cover, validate_cover

class Token(StrEnum):
	T = "T"
	W = "W" 
	Q = "Q" 
	EQUALS = "=" 
	EOS = "E" 
	PAD = "#" 

class Task(MirroredEnum):
	TW_POOL_QUERY = "tw-pool-query"   # [Target-Witness Pool] [Q Q Q ...] = [T T T ...] EOS

@dataclass
class ThreeEdgeOpts(DatasetOpts):
	task: Task
	n_vocab_query: int       # total size of the query vocab 
	n_vocab_witness: int     # total size of the witness vocab 
	n_vocab_target: int      # total size of the target vocab 
	witness_frac: float      # fraction of total witness vocab to be used in TW-pool 
	n_witness_per_link: int  # number of non-zero elements in each (q, p) fiber
	n_target_decoy: int      # number of decoy T in the WT-pool in addition to the target T's
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
	context_len: int = eqx.field(static=True)
	pool_budget: int = eqx.field(static=True)
	s_inds_IWQ: Array

	def __init__(self, opts: ThreeEdgeOpts, is_train: bool, seed: int):
		super().__init__(is_train, opts.train_frac)
		self.opts = opts
		self.seed = seed

		key = jax.random.key(seed)
		W = opts.n_vocab_witness
		Q = opts.n_vocab_query
		T = opts.n_vocab_target

		self.tokens = {
				Token.T: TokenInfo(beg=0, end=T),
				Token.W: TokenInfo(beg=T, end=T+W),
				Token.Q: TokenInfo(beg=T+W, end=T+W+Q),
				Token.EQUALS: TokenInfo(beg=T+W+Q, end=T+W+Q+1),
				Token.EOS: TokenInfo(beg=T+W+Q+1, end=T+W+Q+2),
				Token.PAD: TokenInfo(beg=T+W+Q+2, end=T+W+Q+3),
				}
		choice_fn = lambda key: jax.random.choice(key, T, (opts.n_witness_per_link,), replace=False)
		keys_WQ = jax.random.split(key, num=W*Q).reshape(W, Q)
		self.s_inds_IWQ = jax.vmap(jax.vmap(choice_fn, out_axes=1), out_axes=1)(keys_WQ)
		self.vocab_size = self.tokens[Token.PAD].beg + 1

		# the WT-pool can have up to Q distinct T, while witness_frac controls the
		# fraction of all W that can appear.
		avg_pool_size = W * self.opts.witness_frac + Q

		# in addition to the essential W, there are n_target_decoy decoys 
		self.pool_budget = math.ceil(avg_pool_size * 1.05) + self.opts.n_target_decoy
		self.context_len = self.pool_budget + 2 * Q + 2

	@property
	def run_attrs(self) -> dict[str, Any]:
		return {
				"data_ds_name": "three-edge", 
				"data_ds_task": self.opts.task.value,
				"data_n_witness_per_link": self.opts.n_witness_per_link,
				"data_n_vocab_witness": self.opts.n_vocab_witness,
				"data_n_vocab_query": self.opts.n_vocab_query,
				"data_n_vocab_target": self.opts.n_vocab_target,
				"data_witness_frac": self.opts.witness_frac,
				"data_n_target_decoy": self.opts.n_target_decoy,
				}

	def _generate_pool(self, key: PRNGKeyArray) -> tuple[Array]:
		"""
		Returns:
		mask_W: bool[W] mask defining presence of W tokens in the p-s pool 
		mask_T: bool[T] mask defining presence of T tokens in the p-s pool
		decoy_T: bool[T] mask for any extra T tokens that can be decoys
		found: bool[], whether the solution was found.  (if false, reject)
		"""
		# return w, x, feasible
		Q = self.opts.n_vocab_query
		W = self.opts.n_vocab_witness
		T = self.opts.n_vocab_target
		e_key, pkey, wkey = jax.random.split(key, num=3)
		mask_W = jax.random.bernoulli(pkey, self.opts.witness_frac, W)

		s_inds_IWQ = jnp.where(mask_W[None,:,None], self.s_inds_IWQ, T)
		zeros_TQ = jnp.zeros((T, Q), dtype=jnp.int8)
		marg_TQ = zeros_TQ.at[s_inds_IWQ,jnp.arange(Q)].set(1, mode="drop")

		state, found = exact_cover(e_key, marg_TQ)
		mask_T = state.chosen_R.astype(jnp.bool)
		decoy_T = (state.block_R == 0) & ~mask_T

		# jax.debug.print("is_valid: {}", is_valid)

		# jax.debug.print("mask_T:  {}\n_target_decoy_T: {}\n", 
				  # mask_T.astype(jnp.int32), decoy_T.astype(jnp.int32))

		return mask_W, mask_T, decoy_T, found

	def _compute_target(self, mask_W, mask_T, perm_Q) -> Array:
		"""
		Computes the target values given the WT-pool and given Q permutation.
		Does not check uniqueness
		"""
		s_inds_IWQ = jnp.where(mask_W[None,:,None], self.s_inds_IWQ, -1)
		s_inds_IWQ = jnp.where(mask_T[s_inds_IWQ], s_inds_IWQ, -1)
		return s_inds_IWQ[:,:,perm_Q].max(axis=(0,1))

	def _has_unique_target(self, mask_W, mask_T) -> Array:
		"""
		Return a bool[] denoting whether the target is unique
		"""
		s_inds_IWQ = jnp.where(mask_W[None,:,None], self.s_inds_IWQ, -1)
		s_inds_IWQ = jnp.where(mask_T[s_inds_IWQ], s_inds_IWQ, -1)

		info = jnp.iinfo(s_inds_IWQ.dtype)

		max_inds_Q = jnp.max(s_inds_IWQ, axis=(0,1))
		min_inds_Q = jnp.min(jnp.where(s_inds_IWQ == -1, info.max, s_inds_IWQ), axis=(0,1))
		return jnp.all(max_inds_Q == min_inds_Q)

	@eqx.filter_jit
	def _stats(self, key: PRNGKeyArray, num_trials: int) -> dict[str, Float[Array, ""]]:
		"""
		Returns a dict with:
		avg_num_target -> average number of targets
		avg_num_decoy -> average number of decoys
		avg_num_witness -> average number of witnesses
		fraction_found -> found rate
		"""
		key_N = jax.random.split(key, num=num_trials)
		mask_NW, mask_NT, decoy_NT, found_N = jax.lax.map(
			self._generate_pool, key_N, batch_size=256)
		result = {}
		n_found = found_N.sum()
		result["avg_num_witness"] = jnp.sum(mask_NW * found_N[:,None]) / n_found
		result["avg_num_target"] = jnp.sum(mask_NT * found_N[:,None]) / n_found
		result["avg_num_decoy"] = jnp.sum(decoy_NT * found_N[:,None]) / n_found
		result["fraction_found"] = n_found / num_trials
		return result

	def stats(self, key: PRNGKeyArray, num_trials: int) -> dict[str, float]:
		result = self._stats(key, num_trials)
		return jax.tree.map(lambda a: a.item(), result)

	def _generate_one(self, key: PRNGKeyArray) -> TokensAndProbs:
		"""
		Generate the content of one item 
		obs_sym: [WT-pool] [Q Q Q ...] = [T T T ...] EOS
		input_mask: bool array
		target_code: identifies r.h.s. tokens
		split_hash: hash for splitting into train / test
		"""
		W = self.opts.n_vocab_witness
		Q = self.opts.n_vocab_query
		T = self.opts.n_vocab_target
		L = self.context_len

		keys = jax.random.split(key, num=3)
		mask_W, mask_T, decoy_T, feasible = self._generate_pool(keys[0])

		inds_Q = jax.random.permutation(keys[1], Q)
		inds_T, num_t = jfuncs.partition_masked(jnp.arange(T), mask_T)
		inds_W, num_w = jfuncs.partition_masked(jnp.arange(W), mask_W)
		decoy_inds_T, n_target_decoy = jfuncs.partition_masked(jnp.arange(T), decoy_T)
		
		n_target_decoy = jnp.minimum(n_target_decoy, self.opts.n_target_decoy)

		decoy_inds_T = jfuncs.shuffle_range(keys[2], decoy_inds_T, jnp.int32(0), n_target_decoy)

		in_budget = (num_t + num_w + n_target_decoy <= self.pool_budget)
		active = in_budget & feasible

		# compute the output
		target_Q = self._compute_target(mask_W, mask_T, inds_Q)

		eq_tok = self.tokens[Token.EQUALS].beg
		eos_tok = self.tokens[Token.EOS].beg
		pad_tok = self.tokens[Token.PAD].beg
		t_tok0 = self.tokens[Token.T].beg
		q_tok0 = self.tokens[Token.Q].beg
		w_tok0 = self.tokens[Token.W].beg

		sizes = jnp.array([0, num_t, num_w, n_target_decoy, Q, 1, Q, 1])
		t_beg, w_beg, d_beg, q_beg, eq_beg, t_pred_beg, eos_beg, end = jnp.cumsum(sizes)

		obs_sym = jnp.full(L, pad_tok, dtype=jnp.int32)
		obs_sym = copy_range(obs_sym, inds_T + t_tok0, t_beg, 0, num_t)
		obs_sym = copy_range(obs_sym, inds_W + w_tok0, w_beg, 0, num_w)
		obs_sym = copy_range(obs_sym, decoy_inds_T + t_tok0, d_beg, 0, n_target_decoy)
		obs_sym = copy_range(obs_sym, inds_Q + q_tok0, q_beg)
		obs_sym = obs_sym.at[eq_beg].set(eq_tok)
		obs_sym = copy_range(obs_sym, target_Q + t_tok0, t_pred_beg)
		obs_sym = obs_sym.at[eos_beg].set(eos_tok)
		obs_sym = shuffle_range(keys[3], obs_sym, 0, q_beg) 

		obs_sym = jnp.where(active, obs_sym, jnp.int32(pad_tok))

		input_mask = jnp.arange(L) < eos_beg
		target_code = jnp.full(L, -1, dtype=jnp.int32)
		target_code = copy_range(target_code, jnp.arange(Q), t_beg)

		return TokensAndProbs(
			key=jax.random.key_data(key),
			obs_sym=obs_sym,
			obs_prob=None,
			input_mask=input_mask,
			target_code=target_code,
			active=active,
			split_hash=jfuncs.hash(obs_sym)
		)


	def validate(self, tokens: Array) -> Int[Array, ""]:
		tokens_ok = jnp.all((tokens >= 0) & (tokens < self.vocab_size))
		eq_mask = (tokens == self.tokens[Token.EQUALS].beg)
		eq_ok = eq_mask.sum() == 1
		eq_beg = jnp.argmax(eq_mask) 
		pos = jnp.arange(tokens.size)
		W = self.opts.n_vocab_witness
		Q = self.opts.n_vocab_query
		T = self.opts.n_vocab_target

		def _vocab(code):
			info = self.tokens[code]
			return jnp.arange(info.beg, info.end)

		Q = self.opts.n_vocab_query
		T = Q # we're using each of the Q for the target
		
		q_mask = jnp.isin(tokens, _vocab(Token.Q))
		q_beg = jnp.argmax(q_mask)
		q_end = q_mask.size - jnp.argmax(q_mask[::-1])
		q_ok = (q_mask.sum() == Q) & ((q_end - q_beg) == Q)
		q_vals_T = tokens[q_beg + jnp.arange(Q)] - self.tokens[Token.Q].beg

		t_mask = jnp.isin(jnp.where(pos > eq_beg, tokens, -1), _vocab(Token.T))
		t_beg = jnp.argmax(t_mask)
		t_end = t_mask.size - jnp.argmax(t_mask[::-1])
		t_ok = (t_mask.sum() == Q) & ((t_end - t_beg) == Q)
		t_vals_T = tokens[t_beg + jnp.arange(Q)] - self.tokens[Token.T].beg

		p_active_W = jnp.isin(_vocab(Token.W), jnp.where(pos < eq_beg, tokens, -1))
		s_active_T = jnp.isin(_vocab(Token.T), jnp.where(pos < eq_beg, tokens, -1))

		pred_T = self._compute_target(p_active_W, s_active_T, q_vals_T)
		ans_correct_ok = jnp.all(t_vals_T == pred_T)
		ans_unique_ok = self._has_unique_target(p_active_W, s_active_T)

		status = jnp.array(0, dtype=jnp.int32)
		status = jnp.where(tokens_ok, status, 6)
		status = jnp.where(eq_ok, status, 5)
		status = jnp.where(q_ok, status, 4)
		status = jnp.where(t_ok, status, 3)
		status = jnp.where(ans_correct_ok, status, 2)
		status = jnp.where(ans_unique_ok, status, 1)

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

		pad_idx = list(Token).index(Token.PAD)
		pad_starts = jnp.argmax(tags == pad_idx, axis=1)

		max_dim = max(self.opts.n_vocab_witness, self.opts.n_vocab_query, self.opts.n_vocab_target)

		w = len(str(max_dim))
		out = np.empty((B, L, 2), dtype=f"<U{w}")
		out[:,:,0] = np.array(vals).astype(np.str_)
		out[:,:,1] = codes[tags]

		out = out[item.active] 

		result = []
		for lane, pad_pos in zip(out, pad_starts):
			code_str = " ".join(f"{el:>{w}}" for el in lane[:pad_pos,0])
			vals_str = " ".join(f"{el:>{w}}" for el in lane[:pad_pos,1])
			result.append(code_str + "\n" + vals_str)
		return "\n\n".join(result)



if __name__ == "__main__":
	jnp.set_printoptions(linewidth=205, )
	opts = ThreeEdgeOpts(
		task="three-edge", p=5, q=5, s=100, n_witness_per_link=2, witness_frac=0.8, train_frac=0.75)
	ds = ThreeEdgeDataset(opts, True, 12345)
	key = jax.random.key(93848)
	key_B = jax.random.split(key, num=100)
	mask_W, mask_T, feasible = jax.vmap(ds._generate_pool)(key_B)

	key_B = jax.random.split(key, num=10)
	item = ds._gen_batch(key_B)
	status = ds.validate_item(item)
	result = ds.print_raw_item(item)
	import pdb
	pdb.set_trace()


