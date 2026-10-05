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
from jaxtyping import PRNGKeyArray, Array
from .types import TokensAndProbs
from .. import jfuncs
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

class TokenType(Enum):
	Source = auto()
	Target = auto()
	Value = auto()
	Equals = auto()

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
	def num_target_inds(self):
		if self.rel_target_inds:
			return self.trg_ctx_len * 2 + self.trg_ctx_len
		return self.trg_ctx_len

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

	def encode(self, vals: Array, ty: TokenType) -> Array:
		match ty:
			case TokenType.Source:
				return vals - self.opts.min_source_ind + self.source_idx_token0 
			case TokenType.Target:
				return vals - self.opts.min_target_ind + self.target_idx_token0
			case TokenType.Value:
				return vals + self.value_token0
			case TokenType.Equals:
				return self.equals_token 
			case _:
				raise RuntimeError(f"Unknown ty: {ty}")

	def decode(self, toks: Array, ty: TokenType) -> Array:
		# reverses encode.  But, 
		# tokens are reserved as [*values, source_inds, target_inds, equals]
		match ty:
			case TokenType.Source:
				return toks - self.source_idx_token0 + self.opts.min_source_ind
			case TokenType.Target:
				return toks - self.target_idx_token0 + self.opts.min_target_ind
			case TokenType.Value:
				return toks - self.value_token0
			case TokenType.Equals:
				return toks
			case _:
				raise RuntimeError(f"Unknown ty: {ty}")

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

	def get_source_offsets(self, inds: Array) -> Array:
		if self.opts.rel_source_inds:
			pos = jnp.arange(self.opts.trg_ctx_len) * 2
			return inds - self.opts.src_ctx_len - pos
		return inds

	def get_source_inds(self, offs: Array) -> Array:
		if self.opts.rel_source_inds:
			pos = jnp.arange(self.opts.trg_ctx_len) * 2
			return offs + self.opts.src_ctx_len + pos
		return offs

	def get_target_offsets(self, inds: Array) -> Array:
		if self.opts.rel_target_inds:
			pos = jnp.arange(self.opts.trg_ctx_len) * 2 + 1
			return inds + self.opts.blk_ctx_len - pos
		return inds

	def get_target_inds(self, offs: Array) -> Array:
		if self.opts.rel_target_inds:
			pos = jnp.arange(self.opts.trg_ctx_len) * 2 + 1
			return offs - self.opts.blk_ctx_len + pos
		return offs

	def _generate_one(self, key):

		s_beg, b_beg, e_beg, t_beg = self.sections

		val_key, source_key, target_key = jax.random.split(key, num=3)

		sources = jax.random.choice(val_key, self.opts.num_values, (self.opts.src_ctx_len,))

		source_inds = jax.random.choice(
				source_key, self.opts.src_ctx_len, (self.opts.trg_ctx_len,))
		source_offs = self.get_source_offsets(source_inds)

		target_inds = jax.random.permutation(target_key, self.opts.trg_ctx_len)
		target_offs = self.get_target_offsets(target_inds)

		gathered = sources[source_inds]
		targets = jnp.empty(self.opts.trg_ctx_len, dtype=jnp.int32)
		targets = targets.at[target_inds].set(gathered)

		# jax.debug.print("source_inds: {}", source_inds)
		obs_sym = jnp.empty(self.context_len, dtype=source_offs.dtype)
		obs_sym = obs_sym.at[s_beg:b_beg].set(self.encode(sources, TokenType.Value))
		obs_sym = obs_sym.at[b_beg:e_beg:2].set(self.encode(source_offs, TokenType.Source))
		obs_sym = obs_sym.at[b_beg+1:e_beg:2].set(self.encode(target_offs, TokenType.Target))
		obs_sym = obs_sym.at[e_beg].set(self.encode(None, TokenType.Equals))
		obs_sym = obs_sym.at[t_beg:].set(self.encode(targets, TokenType.Value))

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

	def validate(self, tokens: Array) -> Array:
		s, b, e, t = self.sections
		sources = self.decode(tokens[:b], TokenType.Value) 
		src_offs = self.decode(tokens[b:e:2], TokenType.Source)
		trg_offs = self.decode(tokens[b+1:e:2], TokenType.Target)
		equals = self.decode(tokens[e:t], TokenType.Equals)
		targets = self.decode(tokens[t:], TokenType.Value)

		src_inds = self.get_source_inds(src_offs)
		trg_offs = self.get_target_inds(trg_offs)

		sources_ok = jnp.all((sources >= 0) & (sources < self.opts.num_values))
		targets_ok = jnp.all((targets >= 0) & (targets < self.opts.num_values))
		equal_ok = (equals[0] == self.equals_token)
		copied_ok = jnp.all(targets[trg_offs] == sources[src_inds])

		status = jnp.array(0, dtype=jnp.int32)
		status = jnp.where(copied_ok, status, 4)
		status = jnp.where(equal_ok, status, 3)
		status = jnp.where(targets_ok, status, 2)
		status = jnp.where(sources_ok, status, 1)
		return status

	@eqx.filter_jit
	def _validate_item(self, item: TokensAndProbs) -> Array:
		return jax.vmap(self.validate)(item.obs_sym)

	def validate_item(self, item: TokensAndProbs) -> tuple[bool, list[int]]:
		status_B = self._validate_item(item)  
		pct_active = item.active.sum() / item.active.shape[0]
		if pct_active < self.opts.train_frac * 0.8:
			return False, (
					f"Item had {pct_active} active elements, much less than expected "
					f"{self.opts.train_frac}")

		is_valid_B = jnp.where(item.active, status_B == 0, True)
		all_passed = jnp.all(is_valid_B)

		return all_passed.item(), status_B.tolist()

