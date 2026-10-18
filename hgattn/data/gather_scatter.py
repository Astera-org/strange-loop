"""
A task to implement various routing tasks
"""

from dataclasses import dataclass
import functools
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from typing import Any
from .types import TokensAndProbs
from .. import jfuncs
from . import DatasetOpts
from enum import StrEnum, auto
from ..utils import MirroredEnum
import numpy as np
from numpy.dtypes import StringDType
import string

class Token(StrEnum):
	LEFT = auto()
	RIGHT = auto()
	JUMP = auto()
	SOURCE = auto()
	TARGET = auto()
	EQUALS = auto()
	EOS = auto()

class Task(MirroredEnum):
	COPY = "copy"               # [SOURCES] [L R] [L R] ... [L R] = [TARGETS]
	JUMP_COPY = "jump-copy"     # [SOURCES] [JUMPS] [L R] [L R] ... [L R] = [TARGETS]
	REPORT_PATH = "report-path" # [SOURCES] [JUMPS] [TARGETS = [L R] [L R] [L R] ...
	INFER_INDEX = "infer-index" # [SOURCES] [TARGETS] = [L L L ...]
	THREE_EDGE = "three-edge"   # [S-P Bag] [Q Q Q ...] = [S S S ...]


"""
task invariants:
COPY         : TARGETS[R] = SOURCES[L] for all [L R] 
JUMP_COPY    : TARGETS[R] = SOURCES[JUMPS[L]] for all [L R]
REPORT_PATH  : TARGETS[R] = SOURCES[JUMPS[L]] for all [L R]
INFER_INDEX  : TARGETS[i] = SOURCES[L[i]] for all i (positions in output) 
"""


@dataclass
class GatherScatterOpts(DatasetOpts):
	task: Task
	src_ctx_len: int       # number of context positions in the [sources] section
	trg_ctx_len: int       # number of context positions in the [targets] section
	jmp_ctx_len: int       # number of context positions in the [jumps] section 
	num_values: int        # number of distinct values used for [sources] or [targets]
	rel_offsets: bool      # if set, left, right, jump are relative offsets 
	train_frac: float

	def __post_init__(self):
		if self.task == Task.INFER_INDEX:
			if self.src_ctx_len < self.trg_ctx_len or self.num_values < self.trg_ctx_len:
				raise RuntimeError(
					f"For the infer-index task, src_ctx_len and num_values must be >= trg_ctx_len. "
					f"Got {self.src_ctx_len=}, {self.num_values=}, {self.trg_ctx_len=}")


class GatherScatterDataset(eqx.Module):
	opts: GatherScatterOpts = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	token0: dict[Token, int] = eqx.field(static=True)
	min_offs: dict[Token, int] = eqx.field(static=True)
	num_tokens: dict[Token, int] = eqx.field(static=True)
	layout: dict[Token, slice] = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	pred_ty: Token = eqx.field(static=True)

	def __init__(self, opts: GatherScatterOpts, is_train: bool, seed: int):

		ns = opts.src_ctx_len
		nt = opts.trg_ctx_len
		nj = opts.jmp_ctx_len
		nv = opts.num_values
		
		self.opts = opts
		self.is_train = is_train
		self.seed = seed # currently unused
		self.token0 = {}
		self.min_offs = {}
		self.num_tokens = { Token.SOURCE: nv, Token.EQUALS: 1, Token.EOS: 1 }

		def _cumsum(*vals):
			res = [0]	
			for v in vals:
				res.append(res[-1] + v)
			return tuple(res)

		match opts.task:
			case Task.COPY:
				s, l, e, t, eos, end = _cumsum(ns, 2 * nt, 1, nt, 1)
				self.layout = {
					Token.SOURCE: slice(s, l),
					Token.LEFT: slice(l, e, 2),
					Token.RIGHT: slice(l + 1, e, 2),
					Token.EQUALS: slice(e, t),
					Token.TARGET: slice(t, eos),
					Token.EOS: slice(eos, end),
				}
				self.pred_ty = Token.TARGET
				index_targets = {
					Token.LEFT: Token.SOURCE,
					Token.RIGHT: Token.TARGET,
				}

			case Task.JUMP_COPY:
				s, j, l, e, t, eos, end = _cumsum(ns, nj, 2 * nt, 1, nt, 1)
				self.layout = {
					Token.SOURCE: slice(s, j),
					Token.JUMP: slice(j, l),
					Token.LEFT: slice(l, e, 2),
					Token.RIGHT: slice(l + 1, e, 2),
					Token.EQUALS: slice(e, t),
					Token.TARGET: slice(t, eos),
					Token.EOS: slice(eos, end),
				}
				self.pred_ty = Token.TARGET
				index_targets = {
					Token.LEFT: Token.JUMP,
					Token.JUMP: Token.SOURCE,
					Token.RIGHT: Token.TARGET,
				}
			case Task.INFER_INDEX:
				s, t, e, l, eos, end = _cumsum(ns, nt, 1, nt, 1)
				self.layout = {
					Token.SOURCE: slice(s, t),
					Token.TARGET: slice(t, e),
					Token.EQUALS: slice(e, l),
					Token.LEFT: slice(l, eos),
					Token.EOS: slice(eos, end),
				}
				self.pred_ty = Token.LEFT
				index_targets = {
					Token.LEFT: Token.SOURCE,
				}
			case _:
				raise RuntimeError(f"Unknown task: {task}")

		
		def _max_ind(slc):
			return max(range(*slc.indices(slc.stop)))

		def _num_inds(src, trg):
			if self.opts.rel_offsets:
				max_src = _max_ind(src)
				max_trg = _max_ind(trg)
				inner = src.start - max_trg 
				outer = max_src - trg.start
				return outer - inner + 1
			else:
				return _max_ind(trg) + 1 

		def _min_offset(src, trg):
			if self.opts.rel_offsets:
				return trg.start - _max_ind(src)
			else:
				return 0

		for src, trg in index_targets.items():
			self.num_tokens[src] = _num_inds(self.layout[src], self.layout[trg])
			self.min_offs[src] = _min_offset(self.layout[src], self.layout[trg])

		tok0 = 0
		for ty, ntok in self.num_tokens.items():
			self.token0[ty] = tok0
			tok0 += ntok 
		self.token0[Token.TARGET] = self.token0[Token.SOURCE]

		self.vocab_size = sum(self.num_tokens.values())


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
				return toks + min_off - tok0
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

	def _get_inds_or_offsets(self, tok_ty: Token, do_get_inds: bool, vals: Array) -> Array:

		target_ty = {
			(Task.COPY, Token.LEFT): Token.SOURCE,
			(Task.COPY, Token.RIGHT): Token.TARGET,
			(Task.JUMP_COPY, Token.LEFT): Token.JUMP,
			(Task.JUMP_COPY, Token.RIGHT): Token.TARGET,
			(Task.JUMP_COPY, Token.JUMP): Token.SOURCE,
			(Task.INFER_INDEX, Token.LEFT): Token.SOURCE,
		}[self.opts.task, tok_ty]

		ss = self.layout
		src = ss[tok_ty]
		trg = ss[target_ty]
		src_pos = jnp.arange(src.start, src.stop, src.step)

		if do_get_inds:
			return vals - trg.start + src_pos 
		return vals + trg.start - src_pos

	def get_inds(self, ty: Token, offs: Array) -> Array:
		return self._get_inds_or_offsets(ty, True, offs)

	def get_offsets(self, ty: Token, inds: Array) -> Array:
		return self._get_inds_or_offsets(ty, False, inds)

	def _encode_observed_sym(self, info):
		nt = self.opts.trg_ctx_len
		sym = jnp.empty(self.context_len, dtype=jnp.int32)
		target_code = jnp.full(self.context_len, -1, dtype=jnp.int32)
		target_code = target_code.at[self.layout[self.pred_ty]].set(jnp.arange(nt))
		inp_mask = jnp.full(sym.shape, True)

		def step_fn(sym, item):
			tok_ty, ary, slc = item
			if self.opts.rel_offsets:
				if tok_ty in (Token.LEFT, Token.RIGHT, Token.JUMP):
					ary = self.get_offsets(tok_ty, ary)
			enc = self.encode(ary, tok_ty)
			return sym.at[slc].set(enc)

		items = tuple((ty, ary, self.layout[ty]) for ty, ary in info.items())
		sym = functools.reduce(step_fn, items, sym)

		return sym, target_code, inp_mask

	def _generate_one_copy(self, key):

		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=3)
		info = { Token.EQUALS: None, Token.EOS: None }
		info[Token.SOURCE] = sources = jax.random.choice(keys[0], nv, (ns,))
		info[Token.LEFT] = left = jax.random.choice(keys[1], ns, (nt,))
		info[Token.RIGHT] = right = jax.random.permutation(keys[2], nt)
		gathered = sources[left]
		info[Token.TARGET] = targets = jnp.empty(nt, dtype=jnp.int32).at[right].set(gathered)

		obs_sym, target_code, input_mask = self._encode_observed_sym(info)
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, input_mask, target_code, split_hash

	def _generate_one_jump_copy(self, key):
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nj = self.opts.jmp_ctx_len
		nv = self.opts.num_values

		
		info = { Token.EQUALS: None, Token.EOS: None }
		keys = jax.random.split(key, num=4)

		info[Token.SOURCE] = sources = jax.random.choice(keys[0], nv, (ns,))
		info[Token.JUMP] = jumps = jax.random.choice(keys[1], ns, (nj,)) 
		info[Token.LEFT] = left = jax.random.choice(keys[2], nj, (nt,))
		info[Token.RIGHT] = right = jax.random.permutation(keys[3], nt)

		gathered = sources[jumps[left]]
		info[Token.TARGET] = targets = jnp.empty(nt, dtype=jnp.int32).at[right].set(gathered)

		obs_sym, target_code, input_mask = self._encode_observed_sym(info)
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, input_mask, target_code, split_hash

	def _generate_one_infer_index(self, key):
		ns = self.opts.src_ctx_len
		nt = self.opts.trg_ctx_len
		nv = self.opts.num_values

		keys = jax.random.split(key, num=2)

		info = { Token.EQUALS: None, Token.EOS: None }

		# left indices must not have collisions to ensure deterministic task
		info[Token.SOURCE] = sources = jax.random.permutation(keys[0], nv)[:ns]
		info[Token.LEFT] = left = jax.random.permutation(keys[1], ns)[:nt]
		info[Token.TARGET] = targets = sources[left]

		obs_sym, target_code, input_mask = self._encode_observed_sym(info)
		split_hash = jfuncs.hash(obs_sym)

		return obs_sym, input_mask, target_code, split_hash

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
		return { ty: self.decode(tokens[slc], ty) for ty, slc in self.layout.items() }

	@eqx.filter_jit
	def parse_tokens(self, tokens: Array) -> dict[Token, Array]:
		return eqx.filter_vmap(self._parse_tokens)(tokens)

	def print_raw_item(self, item: TokensAndProbs) -> str:
		alpha = np.array(list(string.ascii_letters + string.digits), dtype="<U1")
		sections = self.parse_tokens(item.obs_sym)
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

		out = np.empty((B, self.context_len), dtype=StringDType())

		for ty, slc in self.layout.items():
			ary = nps[ty]
			out[:,slc] = ary.astype(StringDType())

		result = "\n".join(map(" ".join, out.tolist()))
		return result

	@property
	def run_attrs(self) -> dict[str, Any]:
		return {
				"data_ds_name": "gather-scatter",
				"data_ds_task": self.opts.task.value,
				"data_src_ctx_len": self.opts.src_ctx_len,
				"data_trg_ctx_len": self.opts.trg_ctx_len,
				"data_jmp_ctx_len": self.opts.jmp_ctx_len,
				"data_rel_offsets": self.opts.rel_offsets,
				"data_num_copy_values": self.opts.num_values,
				}


	def validate(self, tokens: Array) -> Array:
		tokens_ok = jnp.all((tokens >= 0) & (tokens < self.vocab_size))

		s = self._parse_tokens(tokens)

		if self.opts.rel_offsets:
			def _adj(tok_ty, ary):
				match tok_ty:
					case Token.LEFT | Token.RIGHT | Token.JUMP:
						return self.get_inds(tok_ty, ary)
					case _:
						return ary
			s = { ty: _adj(ty, ary) for ty, ary in s.items() }

		# jax.debug.breakpoint()

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
		status = jnp.where(tokens_ok, status, 5)
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


def test_gather_scatter():
	ns, nj, nt = 5, 8, 10
	def gen_inds(n, max_val):
		return jnp.arange(n) % max_val

	combos = {
		(Task.COPY, Token.LEFT): jnp.arange(nt) % ns,
		(Task.COPY, Token.RIGHT): jnp.arange(nt) % nt,
		(Task.JUMP_COPY, Token.LEFT): jnp.arange(nt) % nj,
		(Task.JUMP_COPY, Token.RIGHT): jnp.arange(nt) % nt,
		(Task.JUMP_COPY, Token.JUMP): jnp.arange(nj) % ns,
		(Task.INFER_INDEX, Token.LEFT): jnp.arange(nt) % ns,
	}

	for (task, tok_ty), inds in combos.items(): 
		opts = GatherScatterOpts(task, ns, nt, nj, 30, True, 0.75)
		dataset = GatherScatterDataset(opts, True, 12345)
		offs = dataset.get_offsets(tok_ty, inds)
		is_valid = jnp.all(dataset.get_inds(tok_ty, offs) == inds)
		print(f"{opts.task}, {tok_ty}:\n{inds=}\n{offs=}\n{is_valid=}\n")

