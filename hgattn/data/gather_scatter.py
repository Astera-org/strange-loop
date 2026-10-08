"""
A task to implement multiple gather and scatter

[sources] [blk1] [blk2] [blk3] [=] -> [targets]
blk# is [left_idx right_idx]
left_idx must be a valid index into [sources]
right_idx must be a valid index into [targets]
every blk can have any valid left_idx
no two blk's can share the same right_idx
There are N bkl's and matching N targets.

Some variations might include:
	- using relative positions for left_idx and/or right_idx

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
from typing import Any
from .types import TokensAndProbs
from .. import jfuncs
from dataclasses import dataclass
from enum import StrEnum, Enum, auto
import numpy as np
import string

class Token(StrEnum):
	LEFT = auto()
	RIGHT = auto()
	JUMP = auto()
	SOURCE = auto()
	TARGET = auto()
	EQUALS = auto()
	EOS = auto()

class Task(StrEnum):
	COPY = "copy"               # [SOURCES] [L R] [L R] ... [L R] = [TARGETS]
	JUMP_COPY = "jump-copy"     # [SOURCES] [JUMPS] [L R] [L R] ... [L R] = [TARGETS]
	INFER_INDEX = "infer-index" # [SOURCES] [TARGETS] = [L L L ...]

"""
task invariants:
COPY         : TARGETS[R] = SOURCES[L] for all [L R] 
JUMP_COPY    : TARGETS[R] = SOURCES[JUMPS[L]] for all [L R]
INFER_INDEX  : TARGETS[i] = SOURCES[L[i]] for all i (positions in output) 
"""

@dataclass
class GatherScatterOpts:
	task: Task
	src_ctx_len: int       # number of context positions in the [sources] section
	trg_ctx_len: int       # number of context positions in the [targets] section
	jmp_ctx_len: int       # number of context positions in the [jumps] section 
	num_values: int        # number of distinct values used for [sources] or [targets]
	rel_offsets: bool      # if set, left, right, jump are relative offsets 
	train_frac: float

	def __post_init__(self):
		try:
			self.task = Task(self.task)
		except Exception as ex:
			raise RuntimeError(
				f"task invalid.  Must be one of: "
				f"{', '.join(t.value for t in Task)}")


class GatherScatterDataset(eqx.Module):
	opts: GatherScatterOpts = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	token0: dict[Token, int] = eqx.field(static=True)
	min_offs: dict[Token, int] = eqx.field(static=True)

	def __init__(self, opts: GatherScatterOpts, is_train: bool, seed: int):
		self.opts = opts
		self.is_train = is_train
		self.seed = seed # currently unused
		self.min_offs = {}

		ns = opts.src_ctx_len
		nt = opts.trg_ctx_len
		nj = opts.jmp_ctx_len
		nv = opts.num_values
		
		# number of distinct tokens needed for LEFT, RIGHT, and JUMP
		if opts.rel_offsets:
			match opts.task:
				case Task.COPY:
					nleft, nright, njump = ns + 2 * nt, 3 * nt + 1, 0
				case Task.JUMP_COPY:
					nleft, nright, njump = nj + 2 * nt, 3 * nt + 1, ns + nj
				case Task.INFER_INDEX:
					nleft, nright, njump = ns + nt, 0, 0
			self.min_offs[Token.LEFT] = -nleft - 1
			self.min_offs[Token.RIGHT] = 2
			self.min_offs[Token.JUMP] = -njump - 1

		else:
			match opts.task:
				case Task.COPY:
					nleft, nright, njump = ns, nt, 0
				case Task.JUMP_COPY:
					nleft, nright, njump = nj, nt, ns
				case Task.INFER_INDEX:
					nleft, nright, njump = ns, 0, 0
			self.min_offs[Token.LEFT] = 0
			self.min_offs[Token.RIGHT] = 0
			self.min_offs[Token.JUMP] = 0 

		self.token0 = {
				Token.SOURCE: 0,
				Token.TARGET: 0, # same token space
				Token.LEFT: nv,
				Token.RIGHT: nv + nleft,
				Token.JUMP: nv + nleft + nright,
				Token.EQUALS: nv + nleft + nright + njump,
				Token.EOS: nv + nleft + nright + njump + 1,
		}

	@property
	def vocab_size(self):
		nl = self.num_ind_values(Token.LEFT)
		nr = self.num_ind_values(Token.RIGHT)
		nj = self.num_ind_values(Token.JUMP)
		nv = self.opts.num_values
		return {
			Task.COPY: nl + nr + nv + 1,
			Task.JUMP_COPY: ns + nj + nt + 1,
			Task.INFER_INDEX: ns + nv + 1,
		}[self.opts.task]

	def encode(self, vals: Array, ty: Token) -> Array:
		tok0 = self.token0[ty]
		match ty:
			case Token.LEFT | Token.RIGHT | Token.JUMP:
				min_off = self.min_offs[ty]
				return vals - min_off + tok0
			case Token.SOURCE | Token.TARGET:
				return vals + tok0
			case Token.EQUALS | Token.EOS:
				return tok0 
			case _:
				raise RuntimeError(f"Unknown ty: {ty}")

	def decode(self, toks: Array, ty: Token) -> Array:
		tok0 = self.token0[ty]
		match ty:
			case Token.LEFT | Token.RIGHT | Token.JUMP:
				min_off = self.min_offs[ty]
				return toks - tok0 + min_off
			case Token.SOURCE | Token.TARGET:
				return toks - tok0 
			case Token.EQUALS | Token.EOS:
				return toks 
			case _:
				raise RuntimeError(f"Unknown ty: {ty}")

	@property
	def context_len(self):
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len
		match self.opts.task:
			case Task.COPY:
				return ns + 3 * nt + 2
			case Task.JUMP_COPY:
				return ns + nj + 3 * nt + 2
			case Task.INFER_INDEX:
				return ns + 2 * nt + 2

	def get_section_slices(self) -> dict[Token, slice]:
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len
		nv = self.opts.num_values

		def _cumsum(*vals):
			res = [0]	
			for v in vals:
				res.append(res[-1] + v)
			return tuple(res)

		match self.opts.task:
			case Task.COPY:
				s, l, e, t, eos, end = _cumsum(ns, 2 * nt, 1, nt, 1)
				return {
					Token.SOURCE: slice(s, l),
					Token.LEFT: slice(l, e, 2),
					Token.RIGHT: slice(l + 1, e, 2),
					Token.EQUALS: slice(e, t),
					Token.TARGET: slice(t, eos),
					Token.EOS: slice(eos, end),
				}
			case Task.JUMP_COPY:
				s, j, l, e, t, eos, end = _cumsum(ns, nj, 2 * nt, 1, nt, 1)
				return {
					Token.SOURCE: slice(s, j),
					Token.JUMP: slice(j, l),
					Token.LEFT: slice(l, e, 2),
					Token.RIGHT: slice(l + 1, e, 2),
					Token.EQUALS: slice(e, t),
					Token.TARGET: slice(t, eos),
					Token.EOS: slice(eos, end),
				}
			case Task.INFER_INDEX:
				s, t, e, l, eos, end = _cumsum(ns, nt, 1, nt, 1)
				return {
					Token.SOURCE: slice(s, t),
					Token.TARGET: slice(t, e),
					Token.EQUALS: slice(e, l),
					Token.LEFT: slice(l, eos),
					Token.EOS: slice(eos, end),
				}
			case _:
				raise RuntimeError(f"Unknown task: {task}")

	def _get_inds_or_offsets(self, tok_ty: Token, do_get_inds: bool, vals: Array) -> Array:

		target_ty = {
			(Task.COPY, Token.LEFT): Token.SOURCE,
			(Task.COPY, Token.RIGHT): Token.TARGET,
			(Task.JUMP_COPY, Token.LEFT): Token.JUMP,
			(Task.JUMP_COPY, Token.RIGHT): Token.TARGET,
			(Task.JUMP_COPY, Token.JUMP): Token.SOURCE,
			(Task.INFER_INDEX, Token.LEFT): Token.SOURCE,
		}[self.opts.task, tok_ty]

		ss = self.get_section_slices() 
		src = ss[tok_ty]
		trg = ss[target_ty]
		src_pos = jnp.arange(src.start, src.stop, src.step)

		if do_get_inds:
			return vals + trg.start + src_pos 
		return vals + trg.start - src_pos

	def get_inds(self, ty: Token, offs: Array) -> Array:
		return self._get_inds_or_offsets(ty, True, offs)

	def get_offsets(self, ty: Token, inds: Array) -> Array:
		return self._get_inds_or_offsets(ty, False, inds)

	def num_ind_values(self, tok_ty: Token) -> int:
		# compute number of distinct index values for a given token type
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len

		return {
			(Task.COPY, Token.LEFT):       (ns, ns + 2 * nt - 1),
			(Task.COPY, Token.RIGHT):      (ns, ns + 2 * nt),
			(Task.JUMP_COPY, Token.LEFT):  (nj, nj + 2 * nt - 1),
			(Task.JUMP_COPY, Token.RIGHT): (nj, nj + 2 * nt),
			(Task.JUMP_COPY, Token.JUMP):  (ns, ns + nj - 1),
			(Task.INFER_INDEX, Token.LEFT):(ns, ns + nt - 1),
		}[self.opts.task, tok_ty][int(self.opts.rel_offsets)]

	def _generate_one_copy(self, key):

		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=3)
		sources = jax.random.choice(keys[0], nv, (ns,))
		left = jax.random.choice(keys[1], ns, (nt,))
		right = jax.random.permutation(keys[2], nt)
		gathered = sources[left]
		targets = jnp.empty(nt, dtype=jnp.int32)
		targets = targets.at[right].set(gathered)

		# jax.debug.print("left: {}", left)
		sl = self.get_section_slices()

		obs_sym = jnp.empty(self.context_len, dtype=sources.dtype)
		obs_sym = obs_sym.at[sl[Token.SOURCE]].set(self.encode(sources, Token.SOURCE))
		obs_sym = obs_sym.at[sl[Token.LEFT]].set(self.encode(left, Token.LEFT))
		obs_sym = obs_sym.at[sl[Token.RIGHT]].set(self.encode(right, Token.RIGHT))
		obs_sym = obs_sym.at[sl[Token.EQUALS]].set(self.encode(None, Token.EQUALS))
		obs_sym = obs_sym.at[sl[Token.TARGET]].set(self.encode(targets, Token.TARGET))
		obs_sym = obs_sym.at[sl[Token.EOS]].set(self.encode(None, Token.EOS))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[sl[Token.TARGET]].set(jnp.arange(nt))
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, inp_mask, target_code, split_hash

	def _generate_one_jump_copy(self, key):
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=4)

		sources = jax.random.choice(keys[0], nv, (ns,))
		jumps = jax.random.choice(keys[1], ns, (nj,)) 
		left = jax.random.choice(keys[2], nj, (nt,))

		right = jax.random.permutation(keys[3], nt)
		gathered = sources[jumps[left]]
		targets = jnp.empty(nt, dtype=jnp.int32)
		targets = targets.at[right].set(gathered)

		# jax.debug.print("left: {}", left)
		sl = self.get_section_slices()

		obs_sym = jnp.empty(self.context_len, dtype=left.dtype)
		obs_sym = obs_sym.at[sl[Token.SOURCE]].set(self.encode(sources, Token.SOURCE))
		obs_sym = obs_sym.at[sl[Token.JUMP]].set(self.encode(jumps, Token.JUMP))
		obs_sym = obs_sym.at[sl[Token.LEFT]].set(self.encode(left, Token.LEFT))
		obs_sym = obs_sym.at[sl[Token.RIGHT]].set(self.encode(right, Token.RIGHT))
		obs_sym = obs_sym.at[sl[Token.EQUALS]].set(self.encode(None, Token.EQUALS))
		obs_sym = obs_sym.at[sl[Token.TARGET]].set(self.encode(targets, Token.TARGET))
		obs_sym = obs_sym.at[sl[Token.EOS]].set(self.encode(None, Token.EOS))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[sl[Token.TARGET]].set(jnp.arange(nt))
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, inp_mask, target_code, split_hash

	def _generate_one_infer_index(self, key):
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=3)

		sources = jax.random.choice(keys[0], nv, (ns,))
		left = jax.random.choice(keys[1], nj, (nt,))
		targets = sources[left]

		# jax.debug.print("left: {}", left)
		sl = self.get_section_slices()

		obs_sym = jnp.empty(self.context_len, dtype=sources.dtype)
		obs_sym = obs_sym.at[sl[Token.SOURCE]].set(self.encode(sources, Token.SOURCE))
		obs_sym = obs_sym.at[sl[Token.TARGET]].set(self.encode(targets, Token.TARGET))
		obs_sym = obs_sym.at[sl[Token.EQUALS]].set(self.encode(None, Token.EQUALS))
		obs_sym = obs_sym.at[sl[Token.LEFT]].set(self.encode(left, Token.LEFT))
		obs_sym = obs_sym.at[sl[Token.EOS]].set(self.encode(None, Token.EOS))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[sl[Token.LEFT]].set(jnp.arange(nt))
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, inp_mask, target_code, split_hash


	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		gen_fn = { 
			Task.COPY: self._generate_one_copy,
			Task.JUMP_COPY: self._generate_one_jump_copy,
			Task.INFER_INDEX: self._generate_one_infer_index,
			}[self.opts.task]

		obs_sym, inp_mask, target_code, split_hash = gen_fn(key)
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
		item = eqx.filter_vmap(self._gen_one_item)(key_B)
		B = key_B.shape[0]
		train_size = int(B * self.opts.train_frac)
		size = train_size if self.is_train else B - train_size

		def _fraction(x):
			x, _ = jfuncs.partition_masked(x, item.active, 0)
			return x[:size]

		return jax.tree.map(_fraction, item)

	def _parse_tokens(self, tokens: Array) -> dict[Token, Array]:
		# returns decoded sections: sources, left_offs, right_offs, equals, targets
		sl = self.get_section_slices()
		return { ty: self.decode(tokens[slc], ty) for ty, slc in sl.items() }

	@eqx.filter_jit
	def parse_tokens(self, tokens: Array) -> dict[Token, Array]:
		return eqx.filter_vmap(self._parse_tokens)(tokens)

	def print_raw_item(self, item: TokensAndProbs) -> str:
		alpha = np.array(list(string.ascii_letters + string.digits), dtype="<U1")
		sections = self.parse_tokens(item.obs_sym)
		slices = self.get_section_slices()
		nps = {}

		for ty, ary in sections.items():
			match ty:
				case Token.SOURCE | Token.TARGET:
					nps[ty] = alpha[np.asarray(ary) % alpha.size]
				case Token.EOS:
					nps[ty] = np.full(ary.shape, 'EOS')
				case Token.EQUALS:
					nps[ty] = np.full(ary.shape, '=')
				case _:
					nps[ty] = np.asarray(ary)

		B = next(iter(sections.values())).shape[0] 

		out = np.empty((B, self.context_len), dtype=object)

		for ty, slc in slices.items():
			ary = nps[ty]
			out[:,slc] = ary.astype(str)

		out = out.astype(str)
		result = "\n".join(map(" ".join, out.tolist()))
		return result

	@property
	def run_attrs(self) -> dict[str, Any]:
		return {
				"data_ds_name": "gather_scatter",
				"data_src_ctx_len": self.opts.src_ctx_len,
				"data_trg_ctx_len": self.opts.trg_ctx_len,
				"data_rel_offsets": self.opts.rel_offsets,
				"data_num_copy_values": self.opts.num_values,
				}


	def validate(self, tokens: Array) -> Array:
		s = self._parse_tokens(tokens)

		left = s[Token.LEFT]
		right = s.get(Token.RIGHT, None)
		jumps = s.get(Token.JUMP, None)
		sources = s[Token.SOURCE]
		targets = s[Token.TARGET]
		equal = s[Token.EQUALS]
		nv = self.opts.num_values

		sources_ok = jnp.all((sources >= 0) & sources < nv)
		targets_ok = jnp.all((targets >= 0) & targets < nv)
		equal_ok = (equal[0] == self.token0[Token.EQUALS])

		# set sources, targets
		match self.opts.task:
			case Task.COPY:
				copied_ok = jnp.all(targets[right] == sources[left])
			case Task.JUMP_COPY:
				copied_ok = jnp.all(targets[right] == sources[jumps[left]])
			case Task.INFER_INDEX:
				copied_ok = jnp.all(targets == sources[left])

		status = jnp.array(0, dtype=jnp.int32)
		status = jnp.where(copied_ok, status, 4)
		status = jnp.where(equal_ok, status, 3)
		status = jnp.where(targets_ok, status, 2)
		status = jnp.where(sources_ok, status, 1)
		# jax.debug.breakpoint()
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

