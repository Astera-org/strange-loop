import math
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Union
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
	input_beg: int
	input_end: int
	mod_val: int
	use_dpse: bool
	int_base: int|None   # specifies the base for multi-digit integer encoding
	train_frac: float    # fraction in [0, 1] for training split
	split_ty: SplitType  # strategy for train/test split
	task_ty: TaskType    # whether program induction or execution
	min_entropy_frac: float
	poly: polynomial.PolynomialOpts

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

		if self.int_base is None:
			self.int_base = 2**64

	@property
	def max_int_magnitude(self):
		return max(abs(self.input_beg), abs(self.input_end), self.poly.max_int_magnitude)


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

def expand_rpn(
	rpn_code: Array,
	subst_vals: Array,
	sources: Array,
	source_pad_val: int = -1,
	output_pad_val: int = -1
):
	"""
	Replace every occurrence of subst_vals[i] in rpn_code with sources[i] excluding
	padding, while copying all other values verbatim.
	"""
	R = rpn_code.shape[0]
	K, M = sources.shape
	O = sources.size + rpn_code.size

	# rpn_code: [R], subst_vals: [K], sources: [K, M]
	matched_RK = rpn_code[:,None] == subst_vals
	matched_R = jnp.any(matched_RK, axis=-1)
	lookup_R = jnp.argmax(matched_RK, axis=-1)
	merged_RM = jnp.where(matched_R[:,None], sources[lookup_R], rpn_code[:,None])
	is_col0_M = jnp.arange(M) == 0
	mask_RM = jnp.where(matched_R[:,None], merged_RM != source_pad_val, is_col0_M)
	mask_I = mask_RM.ravel()
	mask_idx = jnp.pad(jnp.cumsum(mask_I)[:-1], (1,0), constant_values=0)
	mask_idx = jnp.where(mask_I, mask_idx, O)
	buf = jnp.full((O,), output_pad_val)
	buf = buf.at[mask_idx].set(merged_RM.ravel())
	return buf


class PolySeriesDataset(eqx.Module):
	opts: PolySeriesOpts = eqx.field(static=True)
	pgen: polynomial.PolyGen = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	token_map: dict[str, int] = eqx.field(static=True)
	inv_token_map: list[str] = eqx.field(static=True)
	num_digit_tokens: int = eqx.field(static=True)
	used_int_base: int = eqx.field(static=True)
	rpn_codes: jax.Array
	rpn_input_span: jax.Array # how far back the earliest input goes
	coeff_codes: jax.Array

	def __init__(
		self, 
		opts: PolySeriesOpts,
		is_train: bool, 
		seed: int
	):
		key = jax.random.key(seed)

		self.opts = opts
		self.is_train = is_train

		self.pgen = pg = polynomial.PolyGen(self.opts.poly)
		polys = tuple(pg.generate())
		rpn_codes = tuple(p.to_rpn_code(pg.codes, pg.max_rpn_length) for p in polys)
		rpn_input_span = tuple(p.input_span for p in polys)
		rpn_codes = jnp.array(rpn_codes)
		rpn_input_span = jnp.array(rpn_input_span)
		print(f"Found {len(polys)} distinct polynomial templates")

		key_E = jax.random.split(key, num=rpn_codes.shape[0])
		ent_fn = lambda xs: self._expr_entropy_fraction(*xs, num_trials=1000)
		ent_frac_E = jax.lax.map(ent_fn, (key_E, rpn_codes, rpn_input_span), batch_size=1024)
		active_expr_E = ent_frac_E >= self.opts.min_entropy_frac

		self.rpn_codes = rpn_codes[active_expr_E]
		self.rpn_input_span = rpn_input_span[active_expr_E]
		self.coeff_codes = jnp.array([pg.code_map[c] for c in [pg.const_coeff, *pg.coefficients]])
		print(f"{self.rpn_codes.shape[0]} polynomials passed entropy test")

		if opts.int_base is None:
			raise RuntimeError(f"int_base cannot be None")

		max_output_mag = 2**63 if opts.mod_val is None else opts.mod_val
		max_mag = max(self.opts.max_int_magnitude, max_output_mag)

		self.used_int_base = min(opts.int_base, max_mag)

		D = jfuncs.get_max_digits(max_mag, self.used_int_base)
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
				"+": start_token,
				"-": start_token + 1,
				"=": start_token + 2,
				"0": start_token + 3,
				"PAD": start_token + 3 + self.num_digit_tokens,
				**pg.code_map
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
				self.opts.poly.min_coeff,
				self.opts.poly.max_coeff,
				key1, (self.opts.poly.max_terms,))

		const_coeff = _gen_skip_zero(
				self.opts.poly.min_const_coeff,
				self.opts.poly.max_const_coeff,
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
		rpn_expr: jax.Array,
		rpn_input_span: jax.Array,
		num_trials: int
	) -> jax.Array:
		"""
		Compute average entropy fraction for the `rpn_expr` (plugging in `rpn_consts`
		during the eval).  Evaluate `num_trials` to compute the average.
		"""
		B, I, O = num_trials, self.opts.poly.total_vars, self.opts.n_outputs
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
		rpn_code: jax.Array,
		rpn_input_span: jax.Array,
		rpn_coeffs: jax.Array,
		inputs: jax.Array,
		num_outputs: int
	) -> jax.Array:
		"""
		Evaluate `rpn_code` `num_outputs` times, plugging in `rpn_coeffs` and
		`inputs`.
		"""
		evaluate_fn = partial(
				evaluate_rpn, 
				self.pgen.max_rpn_stack_depth, 
				self.pgen.codes,
				self.opts.mod_val
		)

		def step_fn(state, _):
			variables = state
			next_var = evaluate_fn(rpn_code, rpn_coeffs, variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

		init_state = jnp.roll(inputs, -rpn_input_span)
		_, output = jax.lax.scan(step_fn, init_state, length=num_outputs)
		return output

	@property
	def num_position_bits(self):
		# number of bits reserved for ctx_pos
		return math.ceil(math.log2(self.opts.n_outputs))

	def _generate_one(self, key):
		expr_key, input_key, coeff_key = jax.random.split(key, num=3)

		O = self.opts.n_outputs
		I = self.opts.poly.total_vars
		T = self.opts.poly.max_terms
		E, R = self.rpn_codes.shape

		e = jax.random.choice(expr_key, E)
		rpn_code = self.rpn_codes[e]
		rpn_input_span = self.rpn_input_span[e]
		rpn_coeffs = self.gen_coefficients(coeff_key)

		input_rng = jnp.arange(self.opts.input_beg, self.opts.input_end)
		inputs = jax.random.choice(input_key, input_rng, (I,))
		inputs_mask = jnp.arange(I) < rpn_input_span
		inputs = jnp.where(inputs_mask, inputs, 0)
		outputs = self._evaluate_expr(rpn_code, rpn_input_span, rpn_coeffs, inputs, O)

		def last_found_index(ary, val):
			return jnp.max(jnp.where(ary == val, jnp.arange(ary.shape[0]), -1)) 

		tokenize_opts = (
			self.used_int_base, self.opts.use_dpse, self.token_map["0"], self.token_map["+"],
			self.token_map["-"], self.token_map["PAD"])
		outputs_mask = jnp.full_like(outputs, True)
		coeffs_mask = jnp.full_like(rpn_coeffs, True)
		inputs_enc, inputs_places = jfuncs.tokenize_ints(inputs, inputs_mask, *tokenize_opts)
		outputs_enc, outputs_places = jfuncs.tokenize_ints(outputs, outputs_mask, *tokenize_opts)
		tokenize_fn = lambda v: jfuncs.tokenize_int(v, *tokenize_opts)
		rpn_coeffs_enc = jax.vmap(tokenize_fn)(rpn_coeffs)
		input_logical_sz = last_found_index(inputs_places, rpn_input_span - 1) + 1
		output_logical_sz = last_found_index(outputs_places, O - 1) + 1
		input_sz = inputs_enc.shape[0]
		output_sz = outputs_enc.shape[0]

		obs_sym = jnp.full((R + 1 + input_sz + output_sz,), self.token_map["PAD"], dtype=jnp.int32)
		rpn_tokens = expand_rpn(rpn_code, self.coeff_codes, rpn_coeffs_enc,
						  self.token_map["PAD"], self.token_map["PAD"])

		rpn_size = jnp.argmin(rpn_tokens) # index of first pad

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				"""
				| [RPN_EXPR]  | [=]  | [INPUT] | [OUTPUT] |
				r_beg         e_beg  i_beg       o_beg    sym_end
				"""
				r_beg = 0
				e_beg = rpn_size 
				i_beg = e_beg + 1
				o_beg = i_beg + input_logical_sz 
				sym_end = o_beg + output_logical_sz 
				pred_beg = o_beg

			case TaskType.PROGRAM_INDUCTION:
				"""
				| [INPUT]    | [OUTPUTS] | [=] | [RPN_EXPR] |
				i_beg        o_beg       e_beg r_beg        sym_end
				"""
				i_beg = 0
				o_beg = i_beg + input_logical_sz
				e_beg = o_beg + output_logical_sz 
				r_beg = e_beg + 1
				sym_end = r_beg + rpn_size 
				pred_beg = r_beg
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		obs_sym = jfuncs.copy_range(obs_sym, rpn_tokens, r_beg, 0, rpn_size)
		obs_sym = obs_sym.at[e_beg].set(self.token_map["="])
		obs_sym = jfuncs.copy_range(obs_sym, inputs_enc, i_beg, 0, input_logical_sz)
		obs_sym = jfuncs.copy_range(obs_sym, outputs_enc, o_beg, 0, output_logical_sz)

		inp_mask = jnp.arange(obs_sym.shape[0]) < sym_end

		formula_target = e << self.num_position_bits
		target_code = jnp.full((obs_sym.shape[0],), -1, dtype=jnp.int32)

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				out_code = jnp.where(outputs_places != -1, formula_target + outputs_places, -1)
				target_code = jfuncs.copy_range(target_code, out_code, o_beg, 0, out_code.shape[0]) 
			case TaskType.PROGRAM_INDUCTION:
				out_code = formula_target + jnp.arange(R) 
				out_code = jnp.where(jnp.arange(R) > rpn_size, -1, out_code)
				target_code = jfuncs.copy_range(target_code, out_code, r_beg, 0, out_code.shape[0])
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		match self.opts.split_ty:
			case SplitType.INPUT:
				split_hash = jfuncs.hash(inputs)
			case SplitType.EXPR:
				split_hash = jfuncs.hash(e)
			case SplitType.INPUT_EXPR:
				split_hash = jfuncs.hash(jnp.concatenate((e[None], inputs)))
			case _:
				raise RuntimeError(f"Unrecognized split type: {self.opts.split_ty.value}")

		"""
		jax.debug.print(
				"i_beg: {}\no_beg: {}\ne_beg: {}\nr_beg: {}\nsym_end: {}\n"
				"obs_sym.shape: {}\nrpn_tokens: {}\nequals: {}\n"
				"inputs: {}\ninputs_places: {}\ninputs_enc: {}\n"
				"outputs: {}\noutputs_places: {}\noutputs_enc: {}\n"
				"obs_sym: {}\n",
				i_beg, o_beg, e_beg, r_beg, sym_end,
				obs_sym.shape[0], 
				self.rpn_tokens[e],
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

	def _depad(self, tokens: np.array) -> np.array:
		i = tokens.shape[0] - 1
		while i >= 0:
			if tokens[i] != self.token_map["PAD"]:
				break
			i -= 1
		return tokens[:i+1]

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
		tokens = self._depad(tokens)
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

	def get_target_cat(self, target_code: jax.Array, cat: TargetCategory) -> jax.Array:
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

	def get_target_init(self, cat: TargetCategory) -> jax.Array:
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
	poly_opts = polynomial.PolynomialOpts(
		total_vars=5,
		min_terms=1,
		max_terms=4,
		min_arity=1,
		max_arity=2,
		min_degree=1,
		max_degree=3,
		min_const_coeff=-10,
		max_const_coeff=10,
		min_coeff=-1000,
		max_coeff=1000
	)

	opts = PolySeriesOpts(
		n_outputs=10,
		input_beg=-10,
		input_end=10,
		mod_val=2**16,
		use_dpse=False,
		int_base=100,
		train_frac=0.7,
		split_ty="input",
		task_ty="prog-induction",
		min_entropy_frac=0.9,
		poly=poly_opts
	)

	ds = PolySeriesDataset(opts=opts, is_train=True, seed=9283984)
	gen_key = jax.random.key(42)
	batch_size = 10
	gen_key_B = jax.random.split(gen_key, num=batch_size)
	item = ds._gen_item(gen_key_B)
	item_torch = item.to_torch() # generator is in jax
	tokens = np.array(item.obs_sym)
	for b in range(tokens.shape[0]):
		ds.print_raw(tokens[b])

