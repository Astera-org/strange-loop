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
from ..tools.exact_2dcover import ExactCover2D 
from ..tools import graph

class Token(StrEnum):
	T = "T"
	W = "W" 
	Q = "Q" 
	EQUALS = "=" 
	EOS = "E" 
	PAD = "#" 

class Task(MirroredEnum):
	TW_POOL_QUERY = "tw-pool-query"   # [Target-Witness Pool] [Q Q Q ...] = [T T T ...] EOS
	TW_POOL_DIRECT = "tw-pool-direct" # [Target-Witness Pool] = [T T T ...] EOS
	

@dataclass
class ThreeEdgeOpts(DatasetOpts):
	task: Task
	graph_factored: bool          # whether graph is factored 
	n_vocab_query: int            # total size of the query vocab 
	n_vocab_witness: int          # total size of the witness vocab 
	n_vocab_target: int           # total size of the target vocab 
	n_witness_per_link: int|None  # active when graph_factored=False
	n_witness_per_row: int|None   # graph_factored=True
	n_witness_per_col: int|None   # graph_factored=True
	n_target_decoy: int           # number of decoy T in the WT-pool in addition to the target T's
	max_backtrack_steps: int
	max_propagate_steps: int
	max_search_steps: int
	train_frac: float

	def __post_init__(self):
		if self.graph_factored:
			assert (
					self.n_witness_per_row is not None 
					and self.n_witness_per_col is not None), \
					"n_witness_per_{row,col} must be provided for graph_factored"
		else:
			assert self.n_witness_per_link is not None, \
					"n_witness_per_link must be provided for graph_factored=False"

@dataclass
class TokenInfo:
	beg: int
	end: int


class ThreeEdgeDataset(BaseDataset):
	opts: ThreeEdgeOpts = eqx.field(static=True)
	cover2d: ExactCover2D = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	tokens: dict[Token, TokenInfo] = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	context_len: int = eqx.field(static=True)
	pool_budget: int = eqx.field(static=True)

	def __init__(self, opts: ThreeEdgeOpts, is_train: bool, seed: int):
		super().__init__(is_train, opts.train_frac)
		self.opts = opts
		rng = np.random.default_rng(seed)
		if opts.graph_factored:
			graph_RWC = graph.gen_factored_witness_graph(
				R=opts.n_vocab_target,
				W=opts.n_vocab_witness,
				C=opts.n_vocab_query,
				witness_per_row=opts.n_witness_per_row,
				witness_per_col=opts.n_witness_per_col,
				rng=rng,
			)
		else:
			graph_RWC = graph.gen_witness_graph(
				R=opts.n_vocab_target,
				W=opts.n_vocab_witness,
				C=opts.n_vocab_query,
				n_witness_per_link=opts.n_witness_per_link,
				rng=rng,
			)

		self.cover2d = ExactCover2D(
			jnp.array(graph_RWC),
			extraR=opts.n_target_decoy,
			extraW=0, # for now we don't explore 
			max_backtrack_steps=opts.max_backtrack_steps,
			max_propagate_steps=opts.max_propagate_steps,
			max_search_steps=opts.max_search_steps,
		)

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

		self.vocab_size = self.tokens[Token.PAD].beg + 1

		self.pool_budget = self.cover2d.max_assignment_weight

		match opts.task:
			case Task.TW_POOL_QUERY: 
				self.context_len = self.pool_budget + 2 * Q + 2
			case Task.TW_POOL_DIRECT:
				self.context_len = self.pool_budget + Q + 2
			case _:
				raise RuntimeError(f"Unknown task: {opts.task}")

	@property
	def graph_density(self) -> float:
		return self.cover2d.graph_density.item() 

	@property
	def run_attrs(self) -> dict[str, Any]:
		return {
				"data_ds_name": "three-edge", 
				"data_ds_task": self.opts.task.value,
				"data_graph_factored": self.opts.graph_factored,
				"data_n_vocab_witness": self.opts.n_vocab_witness,
				"data_n_vocab_query": self.opts.n_vocab_query,
				"data_n_vocab_target": self.opts.n_vocab_target,
				"data_n_target_decoy": self.opts.n_target_decoy,
				}

	def _generate_pool(self, key: PRNGKeyArray) -> tuple[Array]:
		"""
		Returns:
		mask_W: defines presence of W tokens in the target-witness pool 
		mask_T: defines presence of T tokens in the target-witness pool
		found: bool[], whether the solution was found.  (if false, reject)
		"""
		res = self.cover2d.search(key)
		mask_W = res.asg.in_W
		mask_T = res.asg.in_R

		return mask_W, mask_T, res.valid

	def _filter_witness_inds(self, mask_W, mask_T) -> Int[Array, "T Q I"]:
		"""
		Returns:
		w_inds_TQI: W indices in the graph filtered by mask_W and mask_T, in [T, Q, I] 
		compact form, with -1 representing NA
		"""
		# query = column, target = row, witness = 'I' (translating betw exact cover
		# and three-edge terminology
		validW_TIQ = self.cover2d.validW_RKC
		indsW_TIQ = self.cover2d.indsW_RKC

		active_TIQ = validW_TIQ & mask_W[indsW_TIQ] & mask_T[:,None,None]
		filt_indsW_TIQ = jnp.where(active_TIQ, indsW_TIQ, -1) 
		return filt_indsW_TIQ.transpose(0,2,1) # TQI

	def _compute_target(self, mask_W, mask_T, perm_Q) -> Array:
		"""
		Computes the target values given the WT-pool and given Q permutation.
		Does not check uniqueness.
		Returns targets_Q[q] = target computed
		"""
		indsW_TQI = self._filter_witness_inds(mask_W, mask_T)
		max_inds_TQ = indsW_TQI.max(axis=2) 
		target_Q = jnp.argmax(max_inds_TQ[:,perm_Q], axis=0) 
		# jax.debug.breakpoint()
		return target_Q

	def _has_unique_target(self, mask_W, mask_T) -> Array:
		"""
		Return a bool[] denoting whether the target is unique
		"""
		indsW_TQI = self._filter_witness_inds(mask_W, mask_T)
		active_TQ = jnp.any(indsW_TQI != -1, axis=2)
		uniq_target_Q = (active_TQ.sum(axis=0) == 1)
		return jnp.all(uniq_target_Q)

	@eqx.filter_jit
	def _stats(self, key: PRNGKeyArray, num_trials: int) -> dict[str, Float[Array, ""]]:
		"""
		Returns a dict with:
		avg_num_target -> average number of targets
		avg_num_witness -> average number of witnesses
		fraction_found -> found rate
		"""
		key_N = jax.random.split(key, num=num_trials)
		mask_NW, mask_NT, found_N = jax.lax.map(self._generate_pool, key_N, batch_size=256)
		result = {}
		n_found = found_N.sum()
		result["avg_num_witness"] = jnp.sum(mask_NW * found_N[:,None]) / n_found
		result["avg_num_target"] = jnp.sum(mask_NT * found_N[:,None]) / n_found
		result["fraction_found"] = n_found / num_trials
		return result

	def stats(self, key: PRNGKeyArray, num_trials: int) -> dict[str, float]:
		result = self._stats(key, num_trials)
		result = jax.tree.map(lambda a: a.item(), result)
		result["max_avail_decoy"] = self.opts.n_target_decoy
		return result

	def graph_repr(self) -> str:
		"""
		Return a human-friendly string representation of the graph
		TODO: retrofit for 
		"""
		inds_QTI = np.array(self.witness_inds_TQI.transpose(1,0,2))
		header = tuple(str(t) for t in ["Q\\T", *range(inds_QTI.shape[1])])
		maxw = max(len(h) for h in header)
		lines = [header]
		for q, row in enumerate(inds_QTI):
			wits = [','.join(str(w) for w in sorted(wits) if w != -1) for wits in row]
			lines.append([str(q)] + wits)
			maxw = max(maxw, *[len(w) for w in wits]) 

		strs = []
		for line in lines:
			str_line = ''.join(f"{el:>{maxw+2}}" for el in line)
			strs.append(str_line)
		return "\n".join(strs)


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

		pool_samp_k, query_perm_k, pool_shuf_k = jax.random.split(key, num=3)
		mask_W, mask_T, found = self._generate_pool(pool_samp_k)

		match self.opts.task:
			case Task.TW_POOL_QUERY:
				inds_Q = jax.random.permutation(query_perm_k, Q)
			case Task.TW_POOL_DIRECT:
				inds_Q = jnp.arange(Q)
			
		inds_T, num_t = jfuncs.partition_masked(jnp.arange(T), mask_T)
		inds_W, num_w = jfuncs.partition_masked(jnp.arange(W), mask_W)
		
		in_budget = (num_t + num_w <= self.pool_budget)
		active = in_budget & found

		# compute the output
		target_Q = self._compute_target(mask_W, mask_T, inds_Q)

		eq_tok = self.tokens[Token.EQUALS].beg
		eos_tok = self.tokens[Token.EOS].beg
		pad_tok = self.tokens[Token.PAD].beg
		t_tok0 = self.tokens[Token.T].beg
		q_tok0 = self.tokens[Token.Q].beg
		w_tok0 = self.tokens[Token.W].beg
		
		match self.opts.task:
			case Task.TW_POOL_QUERY:
				begs = jnp.cumsum(jnp.array([0, num_t, num_w, Q, 1, Q, 1]))
				target_beg, witness_beg, query_beg = begs[:3]
				eq_beg, pred_beg, eos_beg, end = begs[3:] 
			case Task.TW_POOL_DIRECT: 
				begs = jnp.cumsum(jnp.array([0, num_t, num_w, 1, Q, 1]))
				target_beg, witness_beg = begs[:2]
				eq_beg, pred_beg, eos_beg, end = begs[2:] 

		obs_sym = jnp.full(L, pad_tok, dtype=jnp.int32)
		obs_sym = copy_range(obs_sym, inds_T + t_tok0, target_beg, 0, num_t)
		obs_sym = copy_range(obs_sym, inds_W + w_tok0, witness_beg, 0, num_w)
		obs_sym = obs_sym.at[eq_beg].set(eq_tok)
		obs_sym = copy_range(obs_sym, target_Q + t_tok0, pred_beg)
		obs_sym = obs_sym.at[eos_beg].set(eos_tok)

		match self.opts.task:
			case Task.TW_POOL_QUERY:
				obs_sym = copy_range(obs_sym, inds_Q + q_tok0, query_beg)
				obs_sym = shuffle_range(pool_shuf_k, obs_sym, 0, query_beg) 
			case Task.TW_POOL_DIRECT:
				obs_sym = shuffle_range(pool_shuf_k, obs_sym, 0, eq_beg) 

		obs_sym = jnp.where(active, obs_sym, jnp.int32(pad_tok))

		input_mask = jnp.arange(L) < eos_beg
		target_code = jnp.full(L, -1, dtype=jnp.int32)
		target_code = copy_range(target_code, jnp.arange(Q), pred_beg)

		return TokensAndProbs(
			key=jax.random.key_data(key),
			obs_sym=obs_sym,
			obs_prob=None,
			input_mask=input_mask,
			target_code=target_code,
			active=active,
			split_hash=jfuncs.hash(obs_sym)
		)

	def validate(self, item: TokensAndProbs) -> Int[Array, ""]:
		tokens = item.obs_sym
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
		
		match self.opts.task:
			case Task.TW_POOL_QUERY:
				q_mask = jnp.isin(tokens, _vocab(Token.Q))
				q_beg = jnp.argmax(q_mask)
				q_end = q_mask.size - jnp.argmax(q_mask[::-1])
				q_ok = (q_mask.sum() == Q) & ((q_end - q_beg) == Q)
				q_vals_T = tokens[q_beg + jnp.arange(Q)] - self.tokens[Token.Q].beg
			case Task.TW_POOL_DIRECT:
				q_ok = jnp.bool(True)
				q_vals_T = jnp.arange(Q)

		t_mask = jnp.isin(jnp.where(pos > eq_beg, tokens, -1), _vocab(Token.T))
		t_beg = jnp.argmax(t_mask)
		t_end = t_mask.size - jnp.argmax(t_mask[::-1])
		t_ok = (t_mask.sum() == Q) & ((t_end - t_beg) == Q)
		t_vals_T = tokens[t_beg + jnp.arange(Q)] - self.tokens[Token.T].beg

		mask_W = jnp.isin(_vocab(Token.W), jnp.where(pos < eq_beg, tokens, -1))
		mask_T = jnp.isin(_vocab(Token.T), jnp.where(pos < eq_beg, tokens, -1))

		pred_T = self._compute_target(mask_W, mask_T, q_vals_T)
		ans_correct_ok = jnp.all(t_vals_T == pred_T)
		ans_unique_ok = self._has_unique_target(mask_W, mask_T)

		target_code_ok = jnp.all((item.target_code >= 0) == t_mask)

		codes = jnp.array(
			[tokens_ok, eq_ok, q_ok, t_ok, ans_correct_ok, ans_unique_ok, target_code_ok])

		# jax.debug.breakpoint()
		return jnp.where(jnp.all(codes), 0, jnp.argmin(codes)) 

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
		has_pad = jax.vmap(jnp.isin, in_axes=(None,0))(pad_idx, tags)
		pad_starts = jnp.where(has_pad, jnp.argmax(tags == pad_idx, axis=1), L)

		max_dim = max(self.opts.n_vocab_witness, self.opts.n_vocab_query, self.opts.n_vocab_target)

		w = len(str(max_dim))
		out = np.empty((B, L, 2), dtype=f"<U{w}")
		out[:,:,0] = np.array(vals).astype(np.str_)
		out[:,:,1] = codes[tags]

		out = out[item.active] 
		pad_starts = pad_starts[item.active]

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


