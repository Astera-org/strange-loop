"""
A task to implement multiple gather and scatter

[sources] [blk1] [blk2] [blk3] [=] -> [targets]
blk# is [source_idx target_idx]
source_idx must be a valid index into [sources]
target_idx must be a valid index into [targets]
every blk can have any valid source_idx
no two blk's can share the same target_idx
There are N bkl's and matching N targets.

Some variations might include:
	- using relative positions for source_idx and/or target_idx

The S and T tokens may be best allocated separately.
The [sources] and [targets] content obviously come from the same token pool.

sources     blocks                                    targets 
V V V V V V S T S T S T S T S T S T S T S T S T S T = V V V V V V V V V V 
1 2 3 4 5 6 1   2   3   4   5   6   7   8   9   10    1 2 3 4 5 6 7 8 9 10
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from .types import TokensAndProbs
from .. import jfuncs
from dataclasses import dataclass
import numpy as np

@dataclass
class SourceIndex:
	i: int

@dataclass
class TargetIndex:
	i: int

@dataclass
class Value:
	i: int

@dataclass
class Equals:
	pass

Token = SourceIndex | TargetIndex | Value | Equals

@dataclass
class GatherScatterOpts:
	src_ctx_len: int       # number of context positions in the [sources] section
	trg_ctx_len: int       # number of context positions in the [targets] section
	num_values: int        # number of distinct values used for [sources] or [targets]
	rel_source_inds: bool  # if set, pos(source_val) = pos(S) - S, otherwise = S
	rel_target_inds: bool  # if set, pos(target_val) - pos(T) = T.
						   #    otherwise, pos(target_val) - pos(=) - 1 = T
	train_frac: float

	@property
	def num_source_inds(self):
		if self.rel_source_inds:
			return self.trg_ctx_len * 2 + self.src_ctx_len - 1
		return self.src_ctx_len

	@property
	def min_source_ind(self):
		if self.rel_source_inds:
			return - (self.trg_ctx_len * 2 + self.src_ctx_len - 1)
		return 0

	@property
	def min_target_ind(self):
		if self.rel_target_inds:
			return 2 # last block points from itself to one past the equals
		return 0

	@property
	def num_target_inds(self):
		if self.rel_target_inds:
			return self.trg_ctx_len * 2 + trg_ctx_len
		return self.trg_ctx_len

	@property
	def blk_ctx_len(self):
		return self.trg_ctx_len * 2

class GatherScatterDataset(eqx.Module):
	opts: GatherScatterOpts = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	source_idx_token0: int = eqx.field(static=True)
	target_idx_token0: int = eqx.field(static=True)
	value_token0: int = eqx.field(static=True)
	equals_token: int = eqx.field(static=True)

	def __init__(self, opts: GatherScatterOpts, is_train: bool, seed: int):
		self.opts = opts
		self.is_train = is_train
		self.seed = seed # currently unused
		self.value_token0 = 0
		self.source_idx_token0 = self.value_token0 + opts.num_values
		self.target_idx_token0 = self.source_idx_token0 + opts.num_source_inds
		self.equals_token = self.target_idx_token0 + opts.num_target_inds

	@property
	def vocab_size(self):
		return (
				self.opts.num_source_inds + self.opts.num_target_inds +
				self.opts.num_values + 1)

	def decode(self, code: int) -> Token:
		# tokens are reserved as [*values, source_inds, target_inds, equals]
		if code < 0 or code >= self.vocab_size:
			raise RuntimeError(f"Token code {code} not in [0, {self.vocab_size=})")
		if code < self.source_idx_token0:
			return Value(code)
		if code < self.target_idx_token0:
			return SourceIndex(code - self.source_idx_token0 + self.opts.min_source_ind)
		if code < self.equals_token:
			return TargetIndex(code - self.target_idx_token0 + self.opts.min_target_ind)
		return Equals()


	@property
	def context_len(self):
		return self.opts.src_ctx_len + self.opts.trg_ctx_len * 3 + 1

	@property
	def sections(self) -> tuple[int]:
		# return 
		s_beg = 0
		b_beg = self.opts.src_ctx_len
		e_beg = b_beg + self.opts.blk_ctx_len
		t_beg = e_beg + 1 
		return s_beg, b_beg, e_beg, t_beg

	def _generate_one(self, key):

		s_beg, b_beg, e_beg, t_beg = self.sections

		val_key, source_key, target_key = jax.random.split(key, num=3)

		sources = jax.random.choice(val_key, self.opts.num_values, (self.opts.src_ctx_len,))

		source_inds = jax.random.choice(
				source_key, self.opts.num_source_inds, (self.opts.trg_ctx_len,))

		target_inds = jax.random.permutation(target_key, self.opts.trg_ctx_len)

		gathered = sources[source_inds]
		targets = jnp.empty(self.opts.trg_ctx_len, dtype=jnp.int32)
		targets = targets.at[target_inds].set(gathered)

		if self.opts.rel_source_inds:
			source_inds = jnp.arange(self.opts.trg_ctx_len) * 2 - source_inds

		if self.opts.rel_target_inds:
			target_inds = jnp.arange(self.opts.trg_ctx_len) * 2 + target_inds + 1

		obs_sym = jnp.empty(self.context_len, dtype=source_inds.dtype)
		obs_sym = obs_sym.at[s_beg:b_beg].set(sources + self.value_token0)
		obs_sym = obs_sym.at[b_beg:e_beg:2].set(source_inds + self.source_idx_token0)
		obs_sym = obs_sym.at[b_beg+1:e_beg:2].set(target_inds + self.target_idx_token0)
		obs_sym = obs_sym.at[e_beg].set(self.equals_token)
		obs_sym = obs_sym.at[t_beg:].set(targets + self.value_token0)

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[t_beg:].set(jnp.arange(self.opts.trg_ctx_len))
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, inp_mask, target_code, split_hash

	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		obs_sym, inp_mask, target_code, split_hash = self._generate_one(key)
		is_train_frac = (split_hash % 1048576) < int(self.opts.train_frac * 1048576)
		is_active = (self.is_train == is_train_frac)

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
		item = jax.vmap(self._gen_one_item)(key_B)
		B = key_B.shape[0]
		train_size = int(B * self.opts.train_frac)
		size = train_size if self.is_train else B - train_size

		def _fraction(x):
			x, _ = jfuncs.partition_masked(x, item.active, 0)
			return x[:size]

		return jax.tree.map(_fraction, item)

	def print_raw(self, tokens: np.array) -> str:
		val_adj = np.full(self.opts.src_ctx_len, self.value_token0)
		blk_adj = np.empty(self.opts.blk_ctx_len, dtype=np.int32)
		blk_adj[::2] = self.source_idx_token0
		blk_adj[1::2] = self.target_idx_token0
		trg_adj = np.full(self.opts.trg_ctx_len, self.value_token0)

		adj = np.concatenate((val_adj, blk_adj, np.array([0]), trg_adj))
		tokens_adj = tokens - adj
		res = ["=" if tok == self.equals_token else str(tok) for tok in tokens_adj]
		return " ".join(res)

	def print_raw_item(self, item: TokensAndProbs) -> str:
		item = item.to_numpy()
		res = []
		for act, toks in zip(item.active, item.obs_sym):
			if not act:
				continue
			res.append(self.print_raw(toks))
		return "\n".join(res)

	def validate(self, tokens: np.array) -> tuple[bool, str]:
		s, b, e, t = self.sections
		enc = [self.decode(co) for co in tokens]
		sources = enc[:b]
		src_inds = enc[b:e:2]
		trg_inds = enc[b+1:e:2]
		equals = enc[e:t]
		targets = enc[t:]

		source_vals = np.array([s.i for s in sources])
		target_vals = np.array([t.i for t in targets])
		src_ind_vals = np.array([s.i for s in src_inds])
		trg_ind_vals = np.array([t.i for t in trg_inds])

		if not all(isinstance(v, Value) for v in sources):
			return False, f"One or more non-Values in source range"
		if not all(isinstance(si, SourceIndex) for si in src_inds):
			return False, f"One or more non-SourceIndex in source index positions"
		if not all(isinstance(ti, TargetIndex) for ti in trg_inds):
			return False, f"One or more non-TargetIndex in target index positions"
		if not all(isinstance(v, Value) for v in targets):
			return False, f"One or more non-Value in target range"
		if not isinstance(equals[0], Equals):
			return False, f"a non-Equals token in the equals position"
		if not np.all(target_vals[trg_ind_vals] == source_vals[src_ind_vals]):
			return False, f"targets are copied incorrectly"
		return True, "passed"

	def validate_item(self, item: TokensAndProbs) -> tuple[bool, str]:
		"""
		Validate the whole item
		"""
		obs_sym = np.asarray(item.obs_sym)
		active = np.asarray(item.active, dtype=np.bool)

		pct_active = active.sum() / active.shape[0]
		if pct_active < self.opts.train_frac * 0.8:
			return False, (
					f"Item had {pct_active} active elements, much less than expected "
					f"{self.opts.train_frac}")

		all_passed = True
		all_msgs = []
		for b, (toks, act) in enumerate(zip(obs_sym, active)):
			if not act:
				continue
			passed, msg = self.validate(toks)
			all_passed &= passed
			if not passed:
				all_msgs.append(f"batch elem: {b}:\n{msg}")

		return all_passed, "\n".join(all_msgs)





