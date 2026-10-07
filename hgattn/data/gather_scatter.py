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
	Left = auto()
	Right = auto()
	Jump = auto()
	Source = auto()
	Target = auto()
	Equals = auto()
	Eos = auto()

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

	@property
	def num_left_inds(self):
		if self.rel_offsets:
			return self.trg_ctx_len * 2 + self.src_ctx_len - 1
		return self.src_ctx_len

	@property
	def num_right_inds(self):
		if self.rel_offsets:
			return self.trg_ctx_len * 2 + self.trg_ctx_len
		return self.trg_ctx_len

	@property
	def num_jump_inds(self):
		if self.rel_offsets:
			return self.jmp_ctx_len + self.src_ctx_len - 1
		return self.jmp_src_ctx_len

	@property
	def min_left_ind(self):
		if self.rel_offsets:
			return - (self.trg_ctx_len * 2 + self.src_ctx_len - 1)
		return 0

	@property
	def min_right_ind(self):
		if self.rel_offsets:
			return 2 # last block points from itself to one past the equals
		return 0

	@property
	def blk_ctx_len(self):
		return self.trg_ctx_len * 2

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
		
		# number of distinct tokens needed for Left, Right, and Jump
		if opts.rel_offsets:
			match opts.task:
				case Task.COPY:
					nleft, nright, njump = ns + 2 * nt, 3 * nt + 1, 0
				case Task.JUMP_COPY:
					nleft, nright, njump = nj + 2 * nt, 3 * nt + 1, ns + nj
				case Task.INFER_INDEX:
					nleft, nright, njump = ns + nt, 0, 0
			self.min_offs[Token.Left] = -nleft - 1
			self.min_offs[Token.Right] = 2
			self.min_offs[Token.Jump] = -njump - 1

		else:
			match opts.task:
				case Task.COPY:
					nleft, nright, njump = ns, nt, 0
				case Task.JUMP_COPY:
					nleft, nright, njump = nj, nt, ns
				case Task.INFER_INDEX:
					nleft, nright, njump = ns, 0, 0
			self.min_offs[Token.Left] = 0
			self.min_offs[Token.Right] = 0
			self.min_offs[Token.Jump] = 0 

		self.token0 = {
				Token.Source: 0,
				Token.Target: 0, # same token space
				Token.Left: nv,
				Token.Right: nv + nleft,
				Token.Jump: nv + nleft + nright,
				Token.Equals: nv + nleft + nright + njump,
				Token.Eos: nv + nleft + nright + njump + 1,
		}

	@property
	def vocab_size(self):
		ns = self.opts.num_left_inds
		nt = self.opts.num_right_inds
		nj = self.opts.num_jump_inds
		nv = self.opts.num_values

		match self.opts.task:
			case Task.COPY:
				return ns + nt + nv + 1
			case Task.JUMP_COPY:
				return ns + nj + nt + 1
			case Task.INFER_INDEX:
				return ns + nv + 1
			case _:
				raise RuntimeError(f"unknown task: {self.opts.task}")

	def encode(self, vals: Array, ty: Token) -> Array:
		tok0 = self.token0[ty]
		match ty:
			case Token.Left | Token.Right | Token.Jump:
				min_off = self.min_offs[ty]
				return vals - min_off + tok0
			case Token.Source | Token.Target:
				return vals + tok0
			case Token.Equals | Token.Eos:
				return tok0 
			case _:
				raise RuntimeError(f"Unknown ty: {ty}")

	def decode(self, toks: Array, ty: Token) -> Array:
		tok0 = self.token0[ty]
		match ty:
			case Token.Left | Token.Right | Token.Jump:
				min_off = self.min_offs[ty]
				return toks - tok0 + min_off
			case Token.Source | Token.Target:
				return toks - tok0 
			case Token.Equals | Token.Eos:
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
					Token.Source: slice(s, l),
					Token.Left: slice(l, e, 2),
					Token.Right: slice(l + 1, e, 2),
					Token.Equals: slice(e, t),
					Token.Targets: slice(t, eos),
					Token.Eos: slice(eos, end),
				}
			case Task.JUMP_COPY:
				s, j, l, e, t, eos, end = _cumsum(ns, nj, 2 * nt, 1, nt, 1)
				return {
					Token.Source: slice(s, j),
					Token.Jump: slice(j, l),
					Token.Left: slice(l, e, 2),
					Token.Right: slice(l + 1, e, 2),
					Token.Equals: slice(e, t),
					Token.Target: slice(t, eos),
					Token.Eos: slice(eos, end),
				}
			case Task.INFER_INDEX:
				s, t, e, l, eos = _cumsum(ns, nt, 1, nt, 1)
				return {
					Token.Source: slice(s, t),
					Token.Target: slice(t, e),
					Token.Left: slice(s, l, 2),
					Token.Equals: slice(e, e + 1),
					Token.Eos: slice(eos, end),
				}
			case _:
				raise RuntimeError(f"Unknown task: {task}")

	def _get_inds_or_offsets(self, ty: Token, do_get_inds: bool, vals: Array) -> Array:
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len

		def _offs_to_inds(dest_start, offs_start, offs_stride, offs) -> Array:
			# translate offsets into indices
			offs_pos = jnp.arange(offs.shape[0]) * offs_stride + offs_start 
			return offs - offs_pos - dest_start

		def _inds_to_offs(dest_start, inds_start, inds_stride, inds) -> Array:
			inds_pos = jnp.arange(inds.shape[0]) * inds_stride + inds_start
			return inds + dest_start - inds_pos

		args = {
			(Task.COPY, Token.Left):        (0, ns, 2),
			(Task.COPY, Token.Right):       (ns + nt * 2 + 1, ns + 1, 2),
			(Task.JUMP_COPY, Token.Left):   (ns, ns + nj, 2),
			(Task.JUMP_COPY, Token.Right):  (ns, ns + nj + 1, 2),
			(Task.JUMP_COPY, Token.Jump):  (0, ns, 1),
			(Task.INFER_INDEX, Token.Left): (0, ns + nt + 1, 1),
		}[self.opts.task, ty]

		if do_get_inds:
			return _offs_to_inds(*args, vals)
		return _inds_to_offs(*args, vals)

	def get_inds(self, ty: Token, offs: Array) -> Array:
		return self._get_inds_or_offsets(ty, True, offs)

	def get_offsets(self, ty: Token, inds: Array) -> Array:
		return self._get_inds_or_offsets(ty, False, inds)

	def _generate_one_copy(self, key):

		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=3)
		sources = jax.random.choice(keys[0], nv, (ns,))
		left = jax.random.choice(keys[1], ns, (nt,))
		left_offs = self.get_offsets(Token.Left, left)
		right = jax.random.permutation(keys[2], nt)
		right_offs = self.get_offsets(Token.Right, right)
		gathered = sources[left]
		targets = jnp.empty(nt, dtype=jnp.int32)
		targets = targets.at[right].set(gathered)

		# jax.debug.print("left: {}", left)
		s, b, e, t, eos = self.get_sections()
		obs_sym = jnp.empty(self.context_len, dtype=source_offs.dtype)
		obs_sym = obs_sym.at[s:b].set(self.encode(sources, Token.Source))
		obs_sym = obs_sym.at[b:e:2].set(self.encode(left_offs, Token.Left))
		obs_sym = obs_sym.at[b+1:e:2].set(self.encode(right_offs, Token.Right))
		obs_sym = obs_sym.at[e].set(self.encode(None, Token.Equals))
		obs_sym = obs_sym.at[t:eos].set(self.encode(targets, Token.Target))
		obs_sym = obs_sym.at[eos].set(self.encode(None, Token.Eos))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[t:eos].set(jnp.arange(nt))
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
		jump_offs = self.get_offsets(Token.Jump, jumps)
		left = jax.random.choice(keys[2], nj, (nt,))
		left_offs = self.get_offsets(Token.Left, left)

		right = jax.random.permutation(keys[3], nt)
		right_offs = self.get_offsets(Token.Right, right)
		gathered = sources[jumps[left]]
		targets = jnp.empty(nt, dtype=jnp.int32)
		targets = targets.at[right].set(gathered)

		# jax.debug.print("left: {}", left)
		sl = self.get_section_slices()

		obs_sym = jnp.empty(self.context_len, dtype=left_offs.dtype)
		obs_sym = obs_sym.at[sl[Token.Source]].set(self.encode(sources, Token.Source))
		obs_sym = obs_sym.at[sl[Token.Jump]].set(self.encode(jump_offs, Token.Jump))
		obs_sym = obs_sym.at[sl[Token.Left]].set(self.encode(left_offs, Token.Left))
		obs_sym = obs_sym.at[sl[Token.Right]].set(self.encode(right_offs, Token.Right))
		obs_sym = obs_sym.at[sl[Token.Equals]].set(self.encode(None, Token.Equals))
		obs_sym = obs_sym.at[sl[Token.Target]].set(self.encode(targets, Token.Target))
		obs_sym = obs_sym.at[sl[Token.Eos]].set(self.encode(None, Token.Eos))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[sl[Token.Target]].set(jnp.arange(nt))
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
		left_offs =self.get_offsets(Token.Left, left)
		targets = sources[left]

		# jax.debug.print("left: {}", left)
		s, t, e, i, eos = self.get_sections(Task.INFER_INDEX)

		obs_sym = jnp.empty(self.context_len, dtype=source_offs.dtype)
		obs_sym = obs_sym.at[s:t].set(self.encode(sources, Token.Value))
		obs_sym = obs_sym.at[t:e].set(self.encode(targets, Token.Value))
		obs_sym = obs_sym.at[e].set(self.encode(None, Token.Equals))
		obs_sym = obs_sym.at[i:eos].set(self.encode(left_offs, Token.Left))
		obs_sym = obs_sym.at[eos].set(self.encode(None, Token.Eos))

		inp_mask = jnp.full(obs_sym.shape, True)
		target_code = jnp.full(obs_sym.shape, -1, dtype=jnp.int32)
		target_code = target_code.at[i:eos].set(jnp.arange(nt))
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
		alpha = np.array(list(string.printable), dtype="<U1")
		sections = self.parse_tokens(item.obs_sym)
		B, O = left_offs.shape
		offs = jnp.empty((B, 2*O), dtype=left_offs.dtype)
		offs = offs.at[:,::2].set(left_offs)
		offs = offs.at[:,1::2].set(right_offs)

		sources = np.asarray(sources)
		offs = np.asarray(offs)
		equals = np.asarray(equals)
		targets = np.asarray(targets)

		sources_str = alpha[sources % alpha.size]
		targets_str = alpha[targets % alpha.size]

		out = np.concatenate(
				(sources_str, offs.astype(str), np.full((B,1), "="), targets_str), axis=1)

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

		left = self.get_inds(Token.Left, s[Token.Left])
		right = self.get_inds(Token.Right, s[Token.Right]) if Token.Right in s else None
		jumps = self.get_inds(Token.Jump, s[Token.Jump]) if Token.Jump in s else None
		sources = s[Token.Source]
		targets = s[Token.Target]
		equal = s[Token.Equals]
		nv = self.opts.num_values

		import pdb
		pdb.set_trace()

		sources_ok = jnp.all((sources >= 0) & sources < nv)
		targets_ok = jnp.all((targets >= 0) & targets < nv)
		equal_ok = (equal[0] == self.token0[Token.Equals])

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

