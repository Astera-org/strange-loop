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
from ..tools import polynomial
from ..tools.rpn import parse_rpn_value, RPNExpression
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
	min_entropy_frac: float
	total_vars: int
	term_counts: list[int]
	arities: list[int]
	degrees: list[int]

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

def rpn_step(
	global_mod_val: int, 
	codes: tuple[Code],
	state, 
	rpn_token
):
	"""
	Step for scanning an RPN expression.
	"""
	stack, ptr, constants, variables = state

	def push(val):
		return stack.at[ptr].set(val), ptr + 1

	def binary(op_func):
		l, r = stack[ptr-2], stack[ptr-1]
		return stack.at[ptr-2].set(op_func(l, r)), ptr - 1

	def unary(op_func):
		a = stack[ptr-1]
		return stack.at[ptr-1].set(op_func(a)), ptr

	def no_op():
		return stack, ptr

	def mod_fn(func):
		def fn(*args):
			return jnp.mod(func(*args), global_mod_val)
		return fn

	# TODO: perhaps firm this up with polynomial.py
	branches = {
			"NOOP": no_op,
			BinaryOp.ADD: lambda: binary(mod_fn(jnp.add)),
			BinaryOp.MUL: lambda: binary(mod_fn(jnp.multiply)),
			UnaryOp.POW2: lambda: unary(mod_fn(lambda x: jnp.power(x, 2))),
			UnaryOp.POW3: lambda: unary(mod_fn(lambda x: jnp.power(x, 3))),
			}


	for k in codes:
		if isinstance(k, (BinaryOp, UnaryOp)):
			continue
		if k.startswith("c"):
			idx = int(k[1:])
			branches[k] = lambda i=idx: push(constants[i])
		elif k.startswith("x"):
			idx = int(k[1:])
			n = variables.shape[0]
			branches[k] = lambda i=idx: push(variables[n-1-i])
		else:
			pass

	branch_list = [branches[code] for code in codes]

	new_stack, new_ptr = jax.lax.switch(rpn_token, branch_list)
	return (new_stack, new_ptr, constants, variables), None

def evaluate_rpn(
	max_stack_depth: int,
	codes: tuple[Code],
	global_mod_val: int,
	rpn_codes: Array, 
	rpn_consts: Array, 
	variables: Array,
):
	"""
	variables: 
	"""
	stack = jnp.empty((max_stack_depth * 2,), dtype=jnp.int32)
	ptr = jnp.array(0, dtype=jnp.int32)
	state = stack, ptr, rpn_consts, variables
	step_fn = partial(rpn_step, global_mod_val, codes)
	final_state, _ = jax.lax.scan(step_fn, state, rpn_codes)
	final_stack = final_state[0]
	ans = final_stack[0]
	return ans

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
	out, _ = jax.lax.scan(scan_fn, jnp.int32(1), arr)
	return out

def evaluate_poly(
	return_monomials: bool,
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
	factors_t = factors_t.at[term_count:].set(0)
	terms_t = jnp.mod(factors_t * coefficients_t, mod_val)
	out = jnp.mod(terms_t.sum(), mod_val) 
	if return_monomials:
		monomials_m = jnp.zeros(monomials_mv.shape[0], dtype=jnp.int32)
		monomials_m = monomials_m.at[monomial_inds_t].set(terms_t)
		return out, monomials_m
	return out

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
	seed: int = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	token_map: dict[str, int] = eqx.field(static=True)
	inv_token_map: list[str] = eqx.field(static=True)
	num_digit_tokens: int = eqx.field(static=True)
	used_int_base: int = eqx.field(static=True)

	monomials: Array      # i4[m,v] power of variable v in monomial m
	monomial_inds: Array  # i4[p,t] term t in polynomial p is monomials[m]
	term_counts: Array    # i4[p] number of terms in polynomial p
	input_spans: Array    # i4[p] state space size
	expr_codes: Array     # i4[p,e] encoded expressions

	# rpn_codes: Array
	# infix_codes: Array
	# rpn_input_span: Array # how far back the earliest input goes
	# coeff_codes: Array

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
		)

		templates = tuple(pg.templates())
		monomials = pg.monomials()
		self.monomials = jnp.array(monomials) 
		self.monomial_inds = jnp.array([t.monomial_inds for t in templates])
		self.term_counts = jnp.array([t.term_count for t in templates])
		self.input_spans = jnp.array([t.input_span(monomials) for t in templates])
		self.expr_codes = jnp.array([
			t.to_infix_code(monomials, pg.codes, pg.max_infix_length)
			for t in templates])

		"""
		# rpn_codes = tuple(p.to_rpn_code(pg.codes, pg.max_rpn_length) for p in polys)
		# infix_codes = tuple(p.to_infix_code(pg.codes, pg.max_infix_length) for p in polys)
		# rpn_input_span = tuple(p.input_span for p in polys)
		# rpn_codes = jnp.array(rpn_codes)
		# infix_codes = jnp.array(infix_codes)
		# rpn_input_span = jnp.array(rpn_input_span)
		print(f"Found {len(polys)} distinct polynomial templates")

		key_E = jax.random.split(key, num=rpn_codes.shape[0])
		ent_fn = lambda xs: self._expr_entropy_fraction(*xs, num_trials=1000)
		ent_frac_E = jax.lax.map(ent_fn, (key_E, rpn_codes, rpn_input_span), batch_size=1024)
		active_expr_E = ent_frac_E >= self.opts.min_entropy_frac

		self.rpn_codes = rpn_codes[active_expr_E]
		self.infix_codes = infix_codes[active_expr_E]
		self.rpn_input_span = rpn_input_span[active_expr_E]
		self.coeff_codes = jnp.array([pg.code_map[c] for c in [pg.const_coeff, *pg.coefficients]])
		print(f"{self.rpn_codes.shape[0]} polynomials passed entropy test")
		"""

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
		self.inv_token_map = [None] * self.vocab_size
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
				key1, (max(self.opts.term_counts),))

		const_coeff = _gen_skip_zero(
				self.opts.min_const_coeff,
				self.opts.max_const_coeff,
				key2, (1,))
		return jnp.concatenate((const_coeff, coeffs))


	@property
	def num_distinct_output_values(self):
		if self.opts.mod_val is None:
			return 2**32
		return self.opts.mod_val

	@eqx.filter_jit
	def _expr_entropy_fraction(
		self,
		key: PRNGKeyArray,
		rpn_expr: Array,
		rpn_input_span: Array,
		num_trials: int
	) -> Array:
		"""
		Compute average entropy fraction for the `rpn_expr` (plugging in `rpn_consts`
		during the eval).  Evaluate `num_trials` to compute the average.
		"""
		B, I, O = num_trials, self.opts.total_vars, self.opts.n_outputs
		input_key, const_key = jax.random.split(key)

		inputs_BI = jax.random.choice(
			input_key, jnp.arange(self.opts.input_beg, self.opts.input_end), (B, I))

		key_B = jax.random.split(const_key, num=B)
		coeff_BI = jax.vmap(self.gen_coefficients)(key_B)
		
		eval_fn = jax.vmap(self._evaluate_expr, in_axes=(None, None, 0, 0, None))

		outputs_BC = eval_fn(rpn_expr, rpn_input_span, coeff_BI, inputs_BI, O)
		ent_fn = jax.vmap(jfuncs.normalized_entropy, in_axes=(0, None))
		norm_entropy_B = ent_fn(outputs_BC, self.num_distinct_output_values)
		# jax.debug.print("norm_entropy: {}\n", norm_entropy_B.mean())
		return norm_entropy_B.mean()

	def _evaluate_expr(
		self,
		monomial_inds: Array,
		coefficients: Array,
		term_count: Array,
		input_span: Array,
		inputs: Array,
		num_outputs: int
	) -> Array:
		"""
		Evaluate `rpn_code` `num_outputs` times, plugging in `rpn_coeffs` and
		`inputs`.
		"""
		evaluate_fn = partial(
			evaluate_poly, 
			False, 
			self.opts.mod_val, 
			self.monomials,
			term_count,
			monomial_inds,
			coefficients,
		)

		def step_fn(state, _):
			variables = state
			next_var = evaluate_fn(variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

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
		E = self.term_counts.shape[0]

		e = jax.random.choice(expr_key, E)
		coeffs = self.gen_coefficients(coeff_key)

		input_rng = jnp.arange(self.opts.input_beg, self.opts.input_end)
		inputs = jax.random.choice(input_key, input_rng, (I,))
		inputs_mask = jnp.arange(I) < self.input_spans[e] 
		inputs = jnp.where(inputs_mask, inputs, 0)
		outputs = self._evaluate_expr(
			self.monomial_inds[e], 
			coeffs,
			self.term_counts[e],
			self.input_spans[e],
			inputs,
			O)

		def last_found_index(ary, val):
			return jnp.max(jnp.where(ary == val, jnp.arange(ary.shape[0]), -1)) 

		tokenize_opts = (
			self.used_int_base, self.opts.use_dpse, self.token_map["0"], self.token_map["+"],
			self.token_map["-"], self.token_map["PAD"])
		outputs_mask = jnp.full_like(outputs, True)
		coeffs_mask = jnp.full_like(coeffs, True)
		inputs_enc, inputs_places = jfuncs.tokenize_ints(inputs, inputs_mask, *tokenize_opts)
		outputs_enc, outputs_places = jfuncs.tokenize_ints(outputs, outputs_mask, *tokenize_opts)
		tokenize_fn = lambda v: jfuncs.tokenize_int(v, *tokenize_opts)
		coeffs_enc = jax.vmap(tokenize_fn)(coeffs)
		input_logical_sz = last_found_index(inputs_places, input_span - 1) + 1
		output_logical_sz = last_found_index(outputs_places, O - 1) + 1
		input_sz = inputs_enc.shape[0]
		output_sz = outputs_enc.shape[0]

		obs_sym = jnp.full((R + 3 + input_sz + output_sz,), self.token_map["PAD"], dtype=jnp.int32)

		if self.opts.output_infix:
			expr_code = self.infix_codes[e]
		else:
			expr_code = code

		expr_tokens = expand_expr(
			expr_code, self.coeff_codes, coeffs_enc, self.token_map["PAD"],
			self.token_map["PAD"])

		expr_size = jnp.argmin(expr_tokens) # index of first pad

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				"""
				[BOS] | [RPN_EXPR]  | [=]  | [INPUT] | [OUTPUT] | [EOS] |
				      r_beg         e_beg  i_beg       o_beg    t_beg   sym_end
				"""
				r_beg = 1
				e_beg = r_beg + expr_size 
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
				t_beg = r_beg + expr_size 
				sym_end = t_beg + 1 
				pred_beg = r_beg
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		obs_sym = obs_sym.at[0].set(self.token_map["BOS"])
		obs_sym = jfuncs.copy_range(obs_sym, expr_tokens, r_beg, 0, expr_size)
		obs_sym = obs_sym.at[e_beg].set(self.token_map["="])
		obs_sym = jfuncs.copy_range(obs_sym, inputs_enc, i_beg, 0, input_logical_sz)
		obs_sym = jfuncs.copy_range(obs_sym, outputs_enc, o_beg, 0, output_logical_sz)
		obs_sym = obs_sym.at[t_beg].set(self.token_map["EOS"]) 

		inp_mask = jnp.arange(obs_sym.shape[0]) < sym_end

		formula_target = e << self.num_position_bits
		target_code = jnp.full((obs_sym.shape[0],), -1, dtype=jnp.int32)

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				out_code = jnp.where(outputs_places != -1, formula_target + outputs_places, -1)
				target_code = jfuncs.copy_range(target_code, out_code, o_beg, 0, out_code.shape[0]) 
			case TaskType.PROGRAM_INDUCTION:
				out_code = formula_target + jnp.arange(R) 
				out_code = jnp.where(jnp.arange(R) > expr_size, -1, out_code)
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

		"""
		jax.debug.print(
				"i_beg: {}\no_beg: {}\ne_beg: {}\nr_beg: {}\nsym_end: {}\n"
				"obs_sym.shape: {}\nexpr_tokens: {}\nequals: {}\n"
				"inputs: {}\ninputs_places: {}\ninputs_enc: {}\n"
				"outputs: {}\noutputs_places: {}\noutputs_enc: {}\n"
				"obs_sym: {}\n",
				i_beg, o_beg, e_beg, r_beg, sym_end,
				obs_sym.shape[0], 
				self.expr_tokens[e],
				self.token_map["="],
				inputs,
				inputs_places,
				inputs_enc,
				outputs,
				outputs_places,
				outputs_enc,
				obs_sym,
			)
		"""

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

		def _fraction(x):
			x, _ = jfuncs.compact_masked(x, item.active)
			return x[:size]

		item = jax.tree.map(_fraction, item)
		return item

	def print_raw(self, tokens: np.array) -> str:
		res = []
		for tok in tokens.tolist():
			s = self.inv_token_map[tok]
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
				sym = self.inv_token_map[tok]
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
				sym = self.inv_token_map[tok]
				results.append(sym)
		return results

	def decode_tokens(self, tokens: np.array) -> list[int|str]:
		if self.used_int_base is None:
			return self._decode_tokens_no_enc(tokens)
		return self._decode_tokens_enc(tokens)

	def _apply_input_mask(self, item: TokensAndProbs) -> np.ndarray:
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
				return dict(rpn=lhs, vals=rhs)
			case TaskType.PROGRAM_INDUCTION:
				return dict(rpn=rhs, vals=lhs)
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

	def validate(self, tokens: np.array) -> tuple[bool, str]:
		tokens = self._strip_control_tokens(tokens)
		parts = self._split(tokens)
		codes = self.decode_tokens(parts["rpn"])
		series = self.decode_tokens(parts["vals"])
		rpn_vals = [parse_rpn_value(co) for co in codes]
		expr = RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
		# expect variable names x0, x1, ..., xk
		var_ords = { name: int(name[1:]) for name in expr.variable_names }
		max_ord = max(o + 1 for o in var_ords.values())

		for i in range(len(series) - max_ord):
			inputs = series[i:i+max_ord]
			output = series[i+max_ord]
			binds = { n: inputs[max_ord - 1 - o] for n, o in var_ords.items() }
			ans = expr.evaluate(**binds)
			if ans != output:
				return False, (
					f"{ans=} != series[{i}]={series[i]}, "
					f"{binds=}\n"
					f"{expr=}\n"
					f"{series=}\n")
		return True, (
			f"{expr=}\n"
			f"{rpn_vals=}\n"
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
				all_msgs.append(f"batch elem: {b}: {msg}")

		# Validate target mask

		for b, (toks, rng, act) in enumerate(zip(target_masked, target_rng, active)):
			if not act:
				continue
			rpn = toks[rng[0]:rng[1]]
			if rpn[-1] == self.token_map["EOS"]: # hack
				rpn = rpn[:-1]
			codes = self.decode_tokens(rpn)
			rpn_vals = [parse_rpn_value(co) for co in codes]
			try:
				expr = RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
			except Exception as ex:
				import pdb
				pdb.set_trace()
				all_passed = False
				all_msgs.append(f"batch elem: {b}: bad target mask: {ex}")

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
		min_entropy_frac=0.9,
		total_vars=5,
		term_counts=[1,2,3,4],
		arities=[1,2],
		degrees=[1,2],
	)

	ds = PolySeriesDataset(opts=opts, is_train=True, seed=9283984)
	gen_key = jax.random.key(42)
	batch_size = 10
	gen_key_B = jax.random.split(gen_key, num=batch_size)
	item = ds._gen_item(gen_key_B)
	print(ds.print_raw_item(item))

