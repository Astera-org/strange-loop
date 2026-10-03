import math
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Union, Any
from functools import partial, total_ordering
from jaxtyping import PRNGKeyArray, Array
from enum import Enum
from dataclasses import dataclass

from ..tools.mathops import BinaryOp, UnaryOp
from ..tools import polynomial, linalg
from .. import jfuncs
from .types import TokensAndProbs

class SplitType(Enum):
	INPUT = "input"
	EXPR = "expr"
	INPUT_EXPR = "input-expr"

class TaskType(Enum):
	PROGRAM_EXECUTION = "prog-execution"
	PROGRAM_INDUCTION = "prog-induction"


@total_ordering
class TargetCategory(Enum):
	CTX_POS = "ctx_pos"
	EXPR = "expr"

	def __lt__(self, other):
		if self.__class__ is other.__class__:
			return self.value < other.value
		return NotImplemented

Code = Union[BinaryOp, UnaryOp, str] # str represents variable names


@dataclass
class PolySeriesOpts:
	n_outputs: int
	mod_val: int
	input_beg: int
	input_end: int
	min_const_coeff: int # range to sample the const coefficient
	max_const_coeff: int
	min_coeff: int       # range to sample the non-const coefficient
	max_coeff: int
	use_dpse: bool
	int_base: int|None   # specifies the base for multi-digit integer encoding
	train_frac: float    # fraction in [0, 1] for training split
	split_ty: SplitType  # strategy for train/test split
	task_ty: TaskType    # whether program induction or execution
	output_infix: bool   # whether to output infix format
	total_vars: int
	term_counts: list[int]
	arities: list[int]
	degrees: list[int]
	input_spans: list[int]

	def __post_init__(self):
		try:
			self.split_ty = SplitType(self.split_ty)
		except ValueError as v:
			raise ValueError(
					f"Received split_ty {self.split_ty}.  "
					f"Valid split_ty are {', '.join(s.value for s in SplitType)}") from v

		try:
			self.task_ty = TaskType(self.task_ty)
		except ValueError as v:
			raise ValueError(
					f"Received task_ty {self.task_ty}.  "
					f"Valid task_ty are {', '.join(s.value for s in TaskType)}") from v

		if self.mod_val is None:
			raise ValueError(f"mod_val must be provided")

		def _clamp(val):
			return max(0, min(self.mod_val, val))
	
		self.input_beg = _clamp(self.input_beg)
		self.input_end = _clamp(self.input_end)
		self.min_const_coeff = _clamp(self.min_const_coeff)
		self.max_const_coeff = _clamp(self.max_const_coeff)
		self.min_coeff = _clamp(self.min_coeff)
		self.max_coeff = _clamp(self.max_coeff)

		if self.int_base is None:
			self.int_base = self.mod_val

def mod_power(mod_val, val, power):
	branches = [
		lambda x: 1,
		lambda x: x,
		lambda x: jnp.mod(x * x, mod_val),
		lambda x: jnp.mod(x * jnp.mod(x * x, mod_val), mod_val),
	]
	return jax.lax.switch(power, branches, val)

def reduce_mod_product(mod_val, arr):
	def scan_fn(carry, x):
		return jnp.mod(carry * x, mod_val), None
	out, _ = jax.lax.scan(scan_fn, jnp.ones((), dtype=arr.dtype), arr)
	return out

def evaluate_poly(
	mod_val: int,
	monomials_mv: Array,
	term_count: Array,
	monomial_inds_t: Array,
	coefficients_t: Array,
	inputs_v: Array,
) -> Array:
	"""
	Evaluate the polynomial defined by monomial_inds_t
	t: term index
	v: variable index
	m: monomial index
	"""
	power_fn = partial(mod_power, mod_val)
	power_fn = jax.vmap(jax.vmap(power_fn), in_axes=(None, 0))
	exponents_tv = monomials_mv[monomial_inds_t]
	factors_tv = power_fn(inputs_v, exponents_tv)
	factors_t = jax.vmap(partial(reduce_mod_product, mod_val))(factors_tv)

	idx = jnp.arange(factors_t.shape[0])
	factors_t = jnp.where(idx < term_count, factors_t, 0)
	terms_t = jnp.mod(factors_t * coefficients_t, mod_val)
	return jnp.mod(terms_t.sum(), mod_val) 

def evaluate_poly_with_monomials(
	mod_val: int,
	monomials_mv: Array,
	term_count: Array,
	monomial_inds_t: Array,
	coefficients_t: Array,
	inputs_v: Array,
) -> tuple[Array, Array]:
	"""
	Evaluate the polynomial defined by monomial_inds_t
	t: term index
	v: variable index
	m: monomial index

	Returns output, factors
	"""
	power_fn = partial(mod_power, mod_val)
	power_fn = jax.vmap(jax.vmap(power_fn), in_axes=(None, 0))
	all_factors_tv = power_fn(inputs_v, monomials_mv)
	all_factors_t = jax.vmap(partial(reduce_mod_product, mod_val))(all_factors_tv)
	factors_t = all_factors_t[monomial_inds_t]

	idx = jnp.arange(factors_t.shape[0])
	factors_t = jnp.where(idx < term_count, factors_t, 0)
	terms_t = jnp.mod(factors_t * coefficients_t, mod_val)
	out = jnp.mod(terms_t.sum(), mod_val) 
	return out, all_factors_t


def expand_expression(
	expr_code: Array,
	subst_vals: Array,
	sources: Array,
	source_pad_val: int = -1,
	output_pad_val: int = -1
):
	"""
	Replace every occurrence of subst_vals[i] in expr_code with sources[i] excluding
	padding, while copying all other values verbatim.
	"""
	E = expr_code.shape[0]
	K, M = sources.shape
	O = sources.size + expr_code.size

	# expr_code: [E], subst_vals: [K], sources: [K, M]
	matched_EK = expr_code[:,None] == subst_vals
	matched_E = jnp.any(matched_EK, axis=-1)
	lookup_E = jnp.argmax(matched_EK, axis=-1)
	merged_EM = jnp.where(matched_E[:,None], sources[lookup_E], expr_code[:,None])
	is_col0_M = jnp.arange(M) == 0
	mask_EM = jnp.where(matched_E[:,None], merged_EM != source_pad_val, is_col0_M)
	mask_I = mask_EM.ravel()
	mask_idx = jnp.pad(jnp.cumsum(mask_I)[:-1], (1,0), constant_values=0)
	mask_idx = jnp.where(mask_I, mask_idx, O)
	buf = jnp.full((O,), output_pad_val)
	buf = buf.at[mask_idx].set(merged_EM.ravel())
	return buf


class PolySeriesDataset(eqx.Module):
	opts: PolySeriesOpts = eqx.field(static=True)
	pgen: polynomial.PolyGen = eqx.field(static=True)
	expr_templates: tuple[polynomial.PolyTemplate] = eqx.field(static=True)
	seed: int = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	token_map: dict[str, int] = eqx.field(static=True)
	inv_token_map: dict[int, str] = eqx.field(static=True)
	num_digit_tokens: int = eqx.field(static=True)
	used_int_base: int = eqx.field(static=True)

	monomials: Array      # i4[m,v] power of variable v in monomial m
	monomial_inds: Array  # i4[p,t] term t in polynomial p is monomials[m]
	term_counts: Array    # i4[p] number of terms in polynomial p
	input_spans: Array    # i4[p] state space size
	expr_codes: Array     # i4[p,e] encoded expressions
	coeff_codes: Array    # i4[?]   

	def __init__(
		self, 
		opts: PolySeriesOpts,
		is_train: bool, 
		seed: int
	):
		key = jax.random.key(seed)

		self.seed = seed
		self.opts = opts
		self.is_train = is_train

		self.pgen = pg = polynomial.PolyGen(
			total_vars=self.opts.total_vars,
			term_counts=self.opts.term_counts,
			arities=self.opts.arities,
			degrees=self.opts.degrees,
			input_spans=self.opts.input_spans,
		)

		self.expr_templates = templates = tuple(pg.templates())
		monomials = pg.monomials()
		self.monomials = jnp.array(monomials) 
		self.monomial_inds = jnp.array([t.monomial_inds for t in templates])
		self.term_counts = jnp.array([t.term_count for t in templates])
		self.input_spans = jnp.array([t.input_span(monomials) for t in templates])
		self.expr_codes = jnp.array([
			t.to_infix_code(monomials, pg.codes, pg.max_infix_length)
			for t in templates])

		E = self.monomial_inds.shape[0]
		M = self.monomials.shape[0]
		print(f"Found {E} distinct polynomial templates")
		print(f"Found {M} distinct monomials") 
		key_E = jax.random.split(key, E)
		self.coeff_codes = jnp.array([pg.code_map[c] for c in reversed(pg.coefficients)])

		if opts.int_base is None:
			raise RuntimeError(f"int_base cannot be None")

		self.used_int_base = min(opts.int_base, opts.mod_val)

		D = jfuncs.get_max_digits(opts.mod_val, self.used_int_base)
		if opts.use_dpse:
			self.num_digit_tokens = D * self.used_int_base
		else:
			self.num_digit_tokens = self.used_int_base

		if self.num_digit_tokens > 2**17:
			raise RuntimeError(
				f"Settings result in {self.num_digit_tokens} digit tokens. "
				f"If using high opts.mod_val, set int_base to restrict the vocabulary "
				f"required")

		start_token = len(pg.codes)
		self.token_map = {
				**pg.code_map,
				"+": start_token,
				"-": start_token + 1,
				"=": start_token + 2,
				"0": start_token + 3,
				"BOS": start_token + 3 + self.num_digit_tokens,
				"EOS": start_token + 4 + self.num_digit_tokens,
				"PAD": start_token + 5 + self.num_digit_tokens,
		}
		self.vocab_size = self.token_map["PAD"] + 1
		self.inv_token_map = { k: None for k in range(self.vocab_size) }
		for w, tok in self.token_map.items():
			self.inv_token_map[tok] = w

	def gen_coefficients(self, key: PRNGKeyArray) -> Array:
		def _gen_skip_zero(lo, hi, key, shape):
			if lo <= 0 < hi:
				n_vals = hi - lo
			else:
				n_vals = hi - lo + 1
			vals = jax.random.choice(key, n_vals, shape) + lo
			return jnp.where(vals == 0, vals + 1, vals)

		key1, key2 = jax.random.split(key)

		coeffs = _gen_skip_zero(
				self.opts.min_coeff,
				self.opts.max_coeff,
				key1, (max(self.opts.term_counts) - 1,))

		const_coeff = _gen_skip_zero(
				self.opts.min_const_coeff,
				self.opts.max_const_coeff,
				key2, (1,))
		return jnp.concatenate((const_coeff, coeffs))

	@property
	def num_templates(self):
		return self.monomial_inds.shape[0]


	@property
	def num_distinct_output_values(self):
		if self.opts.mod_val is None:
			return 2**32
		return self.opts.mod_val

	@eqx.filter_jit
	def expr_deterministic_fraction(
		self,
		num_trials: int,
		batch_size: int,
		key: PRNGKeyArray,
		expr_index: Array,
	) -> tuple[Array, Array]:
		"""
		Sample a set of possible inputs for the polynomial expression at `expr_index`
		and generate trajectories of length n_outputs for the polynomial and
		all monomials.  Using Gauss-Jordan elimination, determine whether the
		polynomial coefficients are uniquely determined from this trajectory. 
		
		Return a tuple of i32[]: (num consistent, num unique)
		"""
		B, I, O = num_trials, self.opts.total_vars, self.opts.n_outputs
		input_key, const_key = jax.random.split(key)

		inputs_BI = jax.random.choice(
			input_key, jnp.arange(self.opts.input_beg, self.opts.input_end), (B, I))

		key_B = jax.random.split(const_key, num=B)
		coeff_BI = jax.vmap(self.gen_coefficients)(key_B)

		def recur_fn(coeff_I, inputs_I):
			def step_fn(carry, _):
				variables = carry 
				next_var, factors = evaluate_poly_with_monomials(
					self.opts.mod_val,
					self.monomials,
					self.term_counts[expr_index],
					self.monomial_inds[expr_index],
					coeff_I,
					variables
				)
				new_carry = jnp.roll(variables, -1, 0).at[-1].set(next_var)
				return new_carry, (next_var, factors)

			init_state = jnp.roll(inputs_I, -self.input_spans[expr_index])
			_, (output_O, factors_OM) = jax.lax.scan(step_fn, init_state, length=O)
			return output_O, factors_OM

		def solve_fn(xs):
			factors_TO, out_O = xs
			return linalg.gauss_elimination(factors_TO, out_O, self.opts.mod_val)

		out_BO, factors_BTO = jax.lax.map(
			lambda xs: recur_fn(*xs), (coeff_BI, inputs_BI), batch_size=batch_size)
		result = jax.lax.map(solve_fn, (factors_BTO, out_BO), batch_size=batch_size)
		return result.consistent.sum(), result.unique.sum()

	@eqx.filter_jit
	def _template_stats(
		self,
		key: PRNGKeyArray,
		num_trials: int,
		batch_size: int,
	) -> tuple[Array, Array]:
		E = self.num_templates
		key_E = jax.random.split(key, num=E)
		expr_fn = partial(self.expr_deterministic_fraction, num_trials, batch_size)
		num_consistent_E, num_unique_E = jax.lax.map(
				lambda xs: expr_fn(*xs), (key_E, jnp.arange(E)), batch_size=10)
		return num_consistent_E, num_unique_E

		num_inconsistent = jnp.sum(num_consistent_E != num_trials)

	def print_template_stats(self, key: PRNGKeyArray, num_trials: int, batch_size: int):
		"""
		Print a report on each template uniqueness
		"""
		num_consistent_E, num_unique_E = self._template_stats(key, num_trials, batch_size)
		num_inconsistent = jnp.sum(num_consistent_E != num_trials)
		if num_inconsistent != 0:
			raise RuntimeError(
				f"Error: Found {num_inconsistent} inconsistent polynomial templates")

		print("Mod\t#Term\tDeg\tArity\tSpan\tUniq\t#Trial\t#Tmpl")
		# key is (mod_val, term_count, degree, arity, span)
		stats = {} # key => [num_unique, num_trials, num_templates] 
		for e in range(self.num_templates):
			t = self.expr_templates[e]
			span = self.input_spans[e].item()
			key = self.opts.mod_val, t.term_count, t.degree, t.arity, span
			stats.setdefault(key, [0,0,0]) # 
			stats[key][0] += num_unique_E[e]
			stats[key][1] += num_trials
			stats[key][2] += 1
		for key, (unq, tri, tmpl) in stats.items():
			key_str = "\t".join(str(k) for k in key)
			frac = unq / tri 
			print(f"{key_str}\t{frac:4.3f}\t{tri}\t{tmpl}")


	def _recurrent_eval(
		self,
		monomial_inds: Array,
		coefficients: Array,
		term_count: Array,
		input_span: Array,
		inputs: Array,
		num_outputs: int
	) -> Array:
		"""
		Evaluate the polynomial identified by the monomial_inds and coefficients.
		Assumes that 
		"""
		evaluate_fn = partial(
			evaluate_poly, 
			self.opts.mod_val, 
			self.monomials,
			term_count,
			monomial_inds,
			coefficients,
		)

		def step_fn(carry, _):
			variables = carry 
			next_var = evaluate_fn(variables)
			new_carry = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_carry, next_var

		init_state = jnp.roll(inputs, -input_span)
		_, output = jax.lax.scan(step_fn, init_state, length=num_outputs)
		return output


	@property
	def num_position_bits(self):
		# number of bits reserved for ctx_pos
		return math.ceil(math.log2(self.opts.n_outputs))

	def _generate_one(self, key):
		expr_key, input_key, coeff_key = jax.random.split(key, num=3)

		O = self.opts.n_outputs
		I = self.opts.total_vars
		T = max(self.opts.term_counts)
		P, E = self.expr_codes.shape

		p = jax.random.choice(expr_key, P)
		coeffs = self.gen_coefficients(coeff_key)

		input_rng = jnp.arange(self.opts.input_beg, self.opts.input_end)
		inputs = jax.random.choice(input_key, input_rng, (I,))
		inputs_mask = jnp.arange(I) < self.input_spans[p] 
		inputs = jnp.where(inputs_mask, inputs, 0)
		outputs = self._recurrent_eval(
			self.monomial_inds[p], 
			coeffs,
			self.term_counts[p],
			self.input_spans[p],
			inputs,
			O)

		def last_found_index(ary, val):
			return jnp.max(jnp.where(ary == val, jnp.arange(ary.shape[0]), -1)) 

		tokenize_opts = (
			self.used_int_base, self.opts.mod_val, self.opts.use_dpse,
			self.token_map["0"], self.token_map["+"], self.token_map["-"],
			self.token_map["PAD"])
		outputs_mask = jnp.full_like(outputs, True)
		coeffs_mask = jnp.full_like(coeffs, True)
		inputs_enc, inputs_places = jfuncs.tokenize_ints(*tokenize_opts, inputs, inputs_mask)
		outputs_enc, outputs_places = jfuncs.tokenize_ints(*tokenize_opts, outputs, outputs_mask)
		tokenize_fn = lambda v: jfuncs.tokenize_int(*tokenize_opts, v)
		coeffs_enc = jax.vmap(tokenize_fn)(coeffs)
		input_logical_sz = last_found_index(inputs_places, self.input_spans[p] - 1) + 1
		output_logical_sz = last_found_index(outputs_places, O - 1) + 1
		input_sz = inputs_enc.shape[0]
		output_sz = outputs_enc.shape[0]

		expr_tokens = expand_expression(
			self.expr_codes[p], self.coeff_codes, coeffs_enc, self.token_map["PAD"],
			self.token_map["PAD"])

		"""
		jax.debug.print(
				"inputs: {}\n"
				"outputs: {}\n"
				"monomials:\n{}\n"
				"coeffs: {}\n"
				"coeffs_enc:\n{}\n"
				"expr_tokens: {}\n", 
				inputs, outputs, 
				self.monomials[self.monomial_inds[p]], 
				coeffs, coeffs_enc, expr_tokens)
		"""
		# jax.debug.breakpoint()

		expr_logical_sz = jnp.argmin(expr_tokens) # index of first pad
		expr_sz = expr_tokens.shape[0]

		# 3 extra for BOS, [=] and EOS
		obs_sym = jnp.full((expr_sz + 3 + input_sz + output_sz,), self.token_map["PAD"], dtype=jnp.int32)

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				"""
				[BOS] | [RPN_EXPR]  | [=]  | [INPUT] | [OUTPUT] | [EOS] |
				      r_beg         e_beg  i_beg       o_beg    t_beg   sym_end
				"""
				r_beg = 1
				e_beg = r_beg + expr_logical_sz 
				i_beg = e_beg + 1
				o_beg = i_beg + input_logical_sz 
				t_beg = o_beg + output_logical_sz 
				sym_end = t_beg + 1
				pred_beg = o_beg

			case TaskType.PROGRAM_INDUCTION:
				"""
				[BOS] | [INPUT]    | [OUTPUTS] | [=] | [RPN_EXPR] | [EOS] |
				0     i_beg        o_beg       e_beg r_beg        t_beg   sym_end
				"""
				i_beg = 1
				o_beg = i_beg + input_logical_sz
				e_beg = o_beg + output_logical_sz 
				r_beg = e_beg + 1
				t_beg = r_beg + expr_logical_sz 
				sym_end = t_beg + 1 
				pred_beg = r_beg
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		obs_sym = obs_sym.at[0].set(self.token_map["BOS"])
		obs_sym = jfuncs.copy_range(obs_sym, expr_tokens, r_beg, 0, expr_logical_sz)
		obs_sym = obs_sym.at[e_beg].set(self.token_map["="])
		obs_sym = jfuncs.copy_range(obs_sym, inputs_enc, i_beg, 0, input_logical_sz)
		obs_sym = jfuncs.copy_range(obs_sym, outputs_enc, o_beg, 0, output_logical_sz)
		obs_sym = obs_sym.at[t_beg].set(self.token_map["EOS"]) 

		inp_mask = jnp.arange(obs_sym.shape[0]) < sym_end

		formula_target = p << self.num_position_bits
		target_code = jnp.full((obs_sym.shape[0],), -1, dtype=jnp.int32)

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				out_code = jnp.where(outputs_places != -1, formula_target + outputs_places, -1)
				target_code = jfuncs.copy_range(target_code, out_code, o_beg, 0, out_code.shape[0]) 
			case TaskType.PROGRAM_INDUCTION:
				out_code = formula_target + jnp.arange(E) 
				out_code = jnp.where(jnp.arange(E) > expr_logical_sz, -1, out_code)
				target_code = jfuncs.copy_range(target_code, out_code, r_beg, 0, out_code.shape[0])
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		match self.opts.split_ty:
			case SplitType.INPUT:
				split_hash = jfuncs.hash(inputs)
			case SplitType.EXPR:
				split_hash = jfuncs.hash(expr_tokens)
			case SplitType.INPUT_EXPR:
				split_hash = jfuncs.hash(jnp.concatenate((expr_tokens, inputs)))
			case _:
				raise RuntimeError(f"Unrecognized split type: {self.opts.split_ty.value}")

		return obs_sym, inp_mask, target_code, split_hash 

	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		obs_sym_C, input_mask_C, target_code_C, split_hash = self._generate_one(key)
		# obs_prob_C = jax.nn.one_hot(obs_sym_C, self.vocab_size)
		is_train_frac = (split_hash % 1048576) < int(self.opts.train_frac * 1048576)
		is_active = (self.is_train == is_train_frac)

		return TokensAndProbs(
				key=jax.random.key_data(key), 
				obs_sym=obs_sym_C,
				obs_prob=None,
				input_mask=input_mask_C,
				target_code=target_code_C,
				active=is_active)


	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		item = jax.vmap(self._gen_one_item)(key_B)
		B = key_B.shape[0]
		train_size = int(B * self.opts.train_frac)
		size = train_size if self.is_train else B - train_size

		# start_bos = item.obs_sym[:,0] == self.token_map['BOS']
		# jax.debug.print("All BOS {}", jnp.all(start_bos))

		def _fraction(x):
			x, _ = jfuncs.partition_masked(x, item.active, 0)
			return x[:size]

		item_compact = jax.tree.map(_fraction, item)

		start_bos = item_compact.obs_sym[:,0] == self.token_map['BOS']
		start_bos = jnp.where(item_compact.active, start_bos, True)

		# jax.debug.print("after compaction: All BOS {}", jnp.all(start_bos))
		# jax.debug.breakpoint()
		return item_compact

	def print_raw(self, tokens: np.array) -> str:
		res = []
		for tok in tokens.tolist():
			if tok < 0:
				raise RuntimeError(f"Got negative token value: {tok}")
			s = self.inv_token_map.get(tok, None)
			if s is None:
				s = str(tok - self.token_map["0"])
			res.append(s)
		for i in range(len(res) - 1, -1, -1):
			if res[i] != "PAD":
				break

		return " ".join(res[:i+1])

	def _decode_tokens_enc(self, tokens: np.array) -> list[int|str]:
		"""
		Decodes tokens (the 'obs_sym' field), which contains base-token encoded
		integers and string representations of RPNValue.
		"""
		sign, curval, place = None, None, None
		results = []
		plus = self.token_map["+"]
		minus = self.token_map["-"]
		zero = self.token_map["0"]

		digits = range(zero, zero + self.num_digit_tokens)

		for tok in tokens.tolist():
			if tok in (plus, minus):
				if curval is not None:
					results.append(sign * curval)
				sign = 1 if tok == plus else -1
				curval = 0
				place = 1
			elif tok in digits:
				if curval is None:
					raise RuntimeError(f"Invalid symbol sequence")
				val = tok - zero
				if self.opts.use_dpse:
					_, val = divmod(val, self.used_int_base)
				curval = val * place + curval
				place *= self.used_int_base
			else:
				if curval is not None:
					results.append(sign * curval)
					curval = None
				sym = self.inv_token_map.get(tok, None)
				assert sym is not None, f"token {tok} invalid"
				results.append(sym)

		if curval is not None:
			results.append(sign * curval)

		return results

	def _decode_tokens_no_enc(self, tokens: np.array) -> list[int|str]:
		results = []
		zero = self.zero_token
		digits = range(zero, zero + self.opts.mod_val)
		for tok in tokens.tolist():
			if tok in digits:
				results.append(tok - zero)
			else:
				sym = self.inv_token_map.get(tok, None)
				assert sym is not None, f"token {tok} invalid"
				results.append(sym)
		return results

	def decode_tokens(self, tokens: np.array) -> list[int|str]:
		if self.used_int_base is None:
			return self._decode_tokens_no_enc(tokens)
		return self._decode_tokens_enc(tokens)

	def _apply_input_mask(self, item: TokensAndProbs) -> tuple[np.ndarray, np.ndarray]:
		"""
		Uses the input mask to parse the input tokens (different lengths)
		for each element of the batch
		"""
		pad = self.token_map["PAD"]
		tokens = np.asarray(item.obs_sym)
		input_mask = np.asarray(item.input_mask, dtype=np.bool)
		to_padded = np.where(input_mask, tokens, pad)
		pad_start = np.argmax(to_padded != pad, axis=1)
		pad_end = tokens.shape[1] - np.argmax(to_padded[:,::-1] != pad, axis=1) 
		return to_padded, np.stack((pad_start, pad_end), axis=1) 

	def _apply_target_mask(self, item: TokensAndProbs) -> np.ndarray:
		"""
		Uses the target mask to parse input tokens
		"""
		pad = self.token_map["PAD"]
		tokens = np.asarray(item.obs_sym)
		target_mask = np.asarray(item.target_code != -1, dtype=np.bool)
		to_padded = np.where(target_mask, tokens, pad)
		pad_start = np.argmax(to_padded != pad, axis=1)
		pad_end = tokens.shape[1] - np.argmax(to_padded[:,::-1] != pad, axis=1) 
		return to_padded, np.stack((pad_start, pad_end), axis=1) 

	def _strip_control_tokens(self, tokens: np.array) -> np.array:
		"""
		Assumes a pattern of:
		BOS [content] EOS PAD PAD ...
		Returns [content]
		"""
		if tokens[0] != self.token_map["BOS"]:
			raise RuntimeError(f"First token should be BOS")

		i = tokens.shape[0] - 1
		while i >= 0:
			if tokens[i] != self.token_map["PAD"]:
				break
			i -= 1
		if tokens[i] != self.token_map["EOS"]:
			raise RuntimeError(f"Last token before padding should be EOS")

		return tokens[1:i]

	def _split(self, tokens: np.array) -> dict[str, np.array]:
		inds, = np.nonzero(tokens == self.token_map["="])
		if inds.shape[0] != 1:
			raise RuntimeError(
				"Symbol string must have exactly one 'EQUALS' token.  "
				f"Has {inds.shape[0]}")
		lhs, rhs = tokens[:inds[0]], tokens[inds[0]+1:]
		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION:
				return dict(expr=lhs, vals=rhs)
			case TaskType.PROGRAM_INDUCTION:
				return dict(expr=rhs, vals=lhs)
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

	def validate(self, tokens: np.array) -> tuple[bool, str]:
		try:
			tokens = self._strip_control_tokens(tokens)
			parts = self._split(tokens)
			codes = self.decode_tokens(parts["expr"])
			series = self.decode_tokens(parts["vals"])
		except Exception as ex:
			return False, f"Got exception {ex}"

		# expect variable names x0, x1, ..., xk
		V = self.pgen.total_vars
		xs = self.pgen.variables
		span = max(xs.index(co) for co in codes if type(co) is str) + 1
		# import pdb
		# pdb.set_trace()

		for i in range(len(series) - span):
			inputs = series[i:i+span]
			output = series[i+span]
			bound = [inputs[span-1-xs.index(co)] if type(co) is str else co for co in codes]
			ans = polynomial.evaluate_infix_expression(bound, self.opts.mod_val)
			if ans != output:
				return False, (
					f"{ans=} != {output=}\n"
					f"{codes=}\n"
					f"{series=}\n")
		return True, (
			f"{codes=}\n"
			f"{series=}\n"
		)

	def validate_item(self, item: TokensAndProbs) -> tuple[bool, str]:
		"""
		Validate the whole item
		"""
		active = np.asarray(item.active, dtype=np.bool)
		input_masked, input_rng = self._apply_input_mask(item)
		target_masked, target_rng = self._apply_target_mask(item)

		pct_active = active.sum() / active.shape[0]
		if pct_active < self.opts.train_frac * 0.8:
			return False, (
					f"Item had {pct_active} active elements, much less than expected "
					f"{self.opts.train_frac}")

		all_passed = True
		all_msgs = []
		for b, (toks, rng, act) in enumerate(zip(input_masked, input_rng, active)):
			if not act:
				continue
			passed, msg = self.validate(toks[rng[0]:rng[1]])
			all_passed &= passed
			if not passed:
				all_msgs.append(f"batch elem: {b}:\n{msg}")

		# Validate target mask

		"""
		for b, (toks, rng, act) in enumerate(zip(target_masked, target_rng, active)):
			if not act:
				continue
			expr = toks[rng[0]:rng[1]]
			if expr[-1] == self.token_map["EOS"]: # hack
				expr = expr[:-1]
			codes = self.decode_tokens(expr)
			rpn_vals = [parse_rpn_value(co) for co in codes]
			try:
				expr = RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
			except Exception as ex:
				import pdb
				pdb.set_trace()
				all_passed = False
				all_msgs.append(f"batch elem: {b}: bad target mask: {ex}")
		"""

		return all_passed, "\n".join(all_msgs)

	def print_raw_item(self, item: TokensAndProbs) -> str:
		item = item.to_numpy()
		res = []
		for act, toks in zip(item.active, item.obs_sym):
			if not act:
				continue
			res.append(self.print_raw(toks))
		return "\n".join(res)

	@property
	def run_attrs(self) -> dict[str, Any]:
		"""
		Define run attributes for streamvis visualization, describing this dataset
		"""
		attrs = {
			"data_max_poly_terms": max(self.opts.term_counts),
			"data_n_poly_vars": self.opts.total_vars,
			"data_coeff_n_vals": self.opts.max_coeff - self.opts.min_coeff,
			"data_encode_int_base": self.used_int_base,
			"data_max_poly_deg": max(self.opts.degrees),
			"data_max_poly_arity": max(self.opts.arities),
			"data_mod_val": self.opts.mod_val,
			"data_max_input_span": max(self.opts.input_spans),
			"data_series_output_len": self.opts.n_outputs,
			"data_split_ty": self.opts.split_ty.value,
			# "data_train_seed": self.seed,
		}
		return attrs

	def get_target_cat(self, target_code: Array, cat: TargetCategory) -> Array:
		match cat:
			case TargetCategory.CTX_POS:
				obits = (jnp.uint32(1) << self.num_position_bits) - 1
				ctx_vals = jnp.bitwise_and(obits, target_code)
				return jnp.where(target_code == -1, -1, ctx_vals)
			case TargetCategory.EXPR:
				expr_vals = target_code >> self.num_position_bits
				return jnp.where(target_code == -1, -1, expr_vals)
			case _:
				raise RuntimeError(f"Unrecognized cat: {cat}")

	def get_target_init(self, cat: TargetCategory) -> Array:
		match cat:
			case TargetCategory.CTX_POS:
				return jnp.zeros((self.opts.n_outputs,))
			case TargetCategory.EXPR:
				return jnp.zeros((self.rpn_exprs.shape[0],))
			case _:
				raise RuntimeError(f"Unrecognized cat: {cat}")

	def get_target_label(self, cat: TargetCategory) -> np.array:
		match cat:
			case TargetCategory.CTX_POS:
				return np.arange(self.opts.n_outputs),
			case TargetCategory.EXPR: 
				return np.array([self.print_expr(t) for t in self.rpn_codes])
			case _:
				raise RuntimeError(f"Unrecognized cat: {cat}")

if __name__ == "__main__":
	opts = PolySeriesOpts(
		n_outputs=10,
		mod_val=2**16,
		input_beg=0,
		input_end=10,
		min_const_coeff=0,
		max_const_coeff=10,
		min_coeff=0,
		max_coeff=1000,
		use_dpse=False,
		int_base=100,
		train_frac=0.7,
		split_ty="input",
		task_ty="prog-induction",
		output_infix=True,
		total_vars=5,
		term_counts=[1,2,3,4],
		arities=[1,2],
		degrees=[1,2],
		input_spans=[2],
	)

	ds = PolySeriesDataset(opts=opts, is_train=True, seed=9283984)
	gen_key = jax.random.key(42)
	batch_size = 10
	gen_key_B = jax.random.split(gen_key, num=batch_size)
	item = ds._gen_item(gen_key_B)
	print(ds.print_raw_item(item))

