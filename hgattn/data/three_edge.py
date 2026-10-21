from functools import partial
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
	nz: int
	s_frac: float
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
	graph: Array

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
		choice_fn = lambda key: jax.random.choice(key, S, (opts.nz,), replace=False)
		p, q, _ = jnp.ogrid[:P,:Q,:1]
		inds = jax.vmap(choice_fn)(jax.random.split(key, num=P*Q)).reshape(P, Q, -1)
		self.graph = jnp.zeros((P, Q, S), dtype=jnp.int32).at[p, q, inds].set(1)
		self.vocab_size = self.tokens[Token.PAD].beg + 1

	def _maybe_solve(self, key: PRNGKeyArray) -> tuple[Array]:
		"""
		Returns a representation of triplets (p, q, s) sorted by q in [0, Q)
		q_Q: the q value for triplet j
		s_Q: the s value for triplet j 
		x_S: bool[S]: set of s present in the solution 
		feasible: bool[], whether the solution is feasible.  (if false, do not use)
		"""
		# return w, x, feasible
		P = self.opts.p
		Q = self.opts.q
		S = self.opts.s
		skey, wkey = jax.random.split(key)
		mask_S = jax.random.bernoulli(skey, self.opts.s_frac, S)

		marg_PQ = jnp.einsum('pqs, s -> pq', self.graph, mask_S.astype(jnp.int32))

		# (p, q) choices with a unique s for the particular mask_S 
		valid_PQ = (marg_PQ == 1).astype(jnp.int32) 

		# q has at least one p forming a valid (p, q) pair 
		valid_Q = valid_PQ.sum(axis=0)

		# search for each j to find an i
		search_p_fn = jax.vmap(partial(jnp.searchsorted, side="left"), in_axes=(1, 0))
		cdf_PQ = jnp.cumsum(valid_PQ, axis=0) - 1
		target_Q = jax.random.randint(wkey, shape=(Q,), minval=0, maxval=valid_Q)

		p_Q = search_p_fn(cdf_PQ, target_Q)

		mask_P = jnp.full((P,), False).at[p_Q].set(True)

		reachable_QS = jnp.einsum('pqs, p, s -> qs', self.graph, mask_P, mask_S) > 0
		feasible = (reachable_QS.sum(axis=1) == 1).all()

		chosen_QS = jnp.where(mask_S[None,:], self.graph[p_Q, jnp.arange(Q)], 0)
		s_Q = jnp.argmax(chosen_QS, 1) # 

		# jax.debug.breakpoint()

		return p_Q, s_Q, mask_S, feasible

	def _generate_one(self, key: PRNGKeyArray) -> TokensAndProbs:
		"""
		Generate the content of one item 
		obs_sym: [S-P Bag] [Q Q Q ...] = [S S S ...] EOS
		input_mask: bool array
		target_code: identifies r.h.s. tokens
		split_hash: hash for splitting into train / test
		"""
		P = self.opts.p
		Q = self.opts.q
		S = self.opts.s
		L = S + 3 * P + 2
		keys = jax.random.split(key, num=3)
		p_Q, s_Q, x_S, feasible = self._maybe_solve(keys[0])
		q_Q = jnp.arange(Q)
		q_perm_Q = jax.random.permutation(keys[1], Q)
		s_S, num_s = jfuncs.partition_masked(jnp.arange(S), x_S)

		sizes = jnp.array([0, num_s, P, Q, 1, Q, 1])
		s_beg, p_beg, q_beg, eq_beg, t_beg, eos_beg, end = jnp.cumsum(sizes)

		equals_token = self.tokens[Token.EQUALS].beg
		eos_token = self.tokens[Token.EOS].beg
		pad_token = self.tokens[Token.PAD].beg

		obs_sym = jnp.full(L, pad_token, dtype=jnp.int32)
		obs_sym = jfuncs.copy_range(obs_sym, s_S + self.tokens[Token.S].beg, s_beg)
		obs_sym = jfuncs.copy_range(obs_sym, p_Q + self.tokens[Token.P].beg, p_beg)
		obs_sym = jfuncs.copy_range(obs_sym, q_Q[q_perm_Q] + self.tokens[Token.Q].beg, q_beg)
		obs_sym = obs_sym.at[eq_beg].set(equals_token)
		obs_sym = jfuncs.copy_range(obs_sym, s_Q[q_perm_Q] + self.tokens[Token.S].beg, t_beg)
		obs_sym = obs_sym.at[eos_beg].set(eos_token)
		obs_sym = jfuncs.shuffle_range(keys[2], obs_sym, 0, q_beg) 

		input_mask = jnp.arange(L) < eos_beg
		target_code = jnp.full(L, -1, dtype=jnp.int32)
		target_code = jfuncs.copy_range(target_code, jnp.arange(Q), t_beg)

		return TokensAndProbs(
			key=jax.random.key_data(key),
			obs_sym=obs_sym,
			obs_prob=None,
			input_mask=input_mask,
			target_code=target_code,
			active=feasible,
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

		ps_active_PS = p_active_P[:,None] & s_active_S[None,:]
		ps_active_P1S = ps_active_PS[:,None,:]

		# there exists exactly one edge among present (P,S) pairs for each Q
		graph_PTS = self.graph[:,q_vals_T,:]

		active_graph_PTS = jnp.where(ps_active_P1S, graph_PTS, 0)
		n_ans_T = active_graph_PTS.sum(axis=(0,2))
		ans_unique_ok = jnp.all(n_ans_T == 1)
		
		# grab the diagonal for each Q, T pair 
		match_ans_PT = active_graph_PTS[:,jnp.arange(T),t_vals_T]
		match_ans_T = match_ans_PT.sum(axis=0)
		ans_correct_ok = jnp.all(match_ans_T == 1)

		status = jnp.array(0, dtype=jnp.int32)
		status = jnp.where(tokens_ok, status, 6)
		status = jnp.where(eq_ok, status, 5)
		status = jnp.where(q_ok, status, 4)
		status = jnp.where(t_ok, status, 3)
		status = jnp.where(ans_unique_ok, status, 2)
		status = jnp.where(ans_correct_ok, status, 1)

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

		result = "\n\n".join(
				"\n".join(
					" ".join(f"{el:>{w}}" for el in row) 
					for row in pair) 
				for pair in out)
		return result




if __name__ == "__main__":
	jnp.set_printoptions(linewidth=205, )
	opts = ThreeEdgeOpts(
		task="three-edge", p=5, q=5, s=100, nz=2, x_rate=0.50, train_frac=0.75)
	ds = ThreeEdgeDataset(opts, True, 12345)
	key = jax.random.key(93848)
	key_B = jax.random.split(key, num=100)
	q_P, s_P, mask_S, feasible = jax.vmap(ds._maybe_solve)(key_B)

	key_B = jax.random.split(key, num=10)
	item = ds._gen_batch(key_B)
	status = ds.validate_item(item)
	result = ds.print_raw_item(item)
	import pdb
	pdb.set_trace()


