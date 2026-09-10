import math
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from functools import partial, total_ordering
from jaxtyping import PRNGKeyArray, Array
from enum import Enum
from dataclasses import dataclass

from ..tools import polynomial
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


@dataclass
class PolySeriesOpts:
	n_outputs: int
	input_beg: int
	input_end: int
	mod_val: int
	use_dpse: bool
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


def rpn_step(
	global_mod_val: int, 
	codes: tuple[str],
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
			"ADD": lambda: binary(mod_fn(jnp.add)),
			"MUL": lambda: binary(mod_fn(jnp.multiply)),
			"POW2": lambda: unary(mod_fn(lambda x: jnp.power(x, 2))),
			"POW3": lambda: unary(mod_fn(lambda x: jnp.power(x, 3))),
			}

	for k in codes:
		if k.startswith("c"):
			idx = int(k[1:])
			branches[k] = lambda i=idx: push(constants[i])
		elif k.startswith("v"):
			idx = int(k[1:])
			branches[k] = lambda i=idx: push(variables[i])
		else:
			pass

	branch_list = [branches[code] for code in codes]

	new_stack, new_ptr = jax.lax.switch(rpn_token, branch_list)
	return (new_stack, new_ptr, constants, variables), None

def evaluate_rpn(
	max_stack_depth: int,
	codes: tuple[str],
	global_mod_val: int,
	rpn_codes: Array, 
	rpn_consts: Array, 
	variables: Array,
):
	stack = jnp.empty((max_stack_depth * 2,), dtype=jnp.int32)
	ptr = jnp.array(0, dtype=jnp.int32)
	state = stack, ptr, rpn_consts, variables
	step_fn = partial(rpn_step, global_mod_val, codes)
	final_state, _ = jax.lax.scan(step_fn, state, rpn_codes)
	final_stack = final_state[0]
	ans = final_stack[0]
	return ans

class PolySeriesDataset(eqx.Module):
	opts: PolySeriesOpts = eqx.field(static=True)
	pgen: polynomial.PolyGen = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	rpn_codes: jax.Array
	rpn_input_span: jax.Array # how far back the earliest input goes

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

		key_E = jax.random.split(key, num=rpn_codes.shape[0])
		ent_fn = lambda xs: self._expr_entropy_fraction(*xs, 1000)
		ent_frac_E = jax.lax.map(ent_fn, (key_E, rpn_codes), batch_size=1024)
		active_expr_E = ent_frac_E >= self.opts.min_entropy_frac

		self.rpn_codes = rpn_codes[active_expr_E]
		self.rpn_input_span = rpn_input_span[active_expr_E]


	@eqx.filter_jit
	def _expr_entropy_fraction(
		self,
		key: PRNGKeyArray,
		rpn_expr: jax.Array,
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

		coeff_BI = jax.random.choice(
			const_key, jnp.arange(self.opts.poly.min_coeff, self.opts.poly.max_coeff), (B, I))

		coeff_BI = coeff_BI.at[:,0].set(
				jax.random.choice(
					const_key, jnp.arange(
						self.opts.poly.min_const_coeff,
						self.opts.poly.max_const_coeff), (B,)))

		eval_fn = jax.vmap(self._evaluate_expr, in_axes=(None, 0, 0, None))

		outputs_BC = eval_fn(rpn_expr, coeff_BI, inputs_BI, O)
		ent_fn = jax.vmap(arith.normalized_entropy, in_axes=(0, None))
		norm_entropy_B = ent_fn(outputs_BC, self.num_distinct_output_values)
		# jax.debug.print("norm_entropy: {}\n", norm_entropy_B.mean())
		return norm_entropy_B.mean()

	def _evaluate_expr(
		self,
		rpn_expr: jax.Array,
		rpn_consts: jax.Array,
		inputs: jax.Array,
		num_outputs: int
	) -> jax.Array:
		"""
		Evaluate `rpn_expr` `num_outputs` times, plugging in `rpn_consts` and
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
			next_var = evaluate_fn(rpn_expr, rpn_consts, variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

		_, output = jax.lax.scan(step_fn, inputs, length=num_outputs)
		return output

	@property
	def num_position_bits(self):
		# number of bits reserved for ctx_pos
		return math.ceil(math.log2(self.opts.n_outputs))


	def _generate_one(self, key):
		expr_key, input_key = jax.random.split(key)

		O = self.opts.n_outputs
		I = self.opts.poly.total_vars
		E, R = self.rpn_tokens.shape

		e = jax.random.choice(expr_key, E)
		rpn_expr = self.rpn_exprs[e]
		rpn_degree = self.rpn_degree[e]
		coeffs_rng = jnp.arange(self.opts.poly.min_coeff, self.opts.poly.max_coeff)
		rpn_coeffs = jax.random.choice(coeff_key, coeffs_rng, (self.opts.poly.max_terms,))

		input_rng = jnp.arange(self.opts.input_beg, self.opts.input_end)
		inputs = jax.random.choice(input_key, input_rng, (I,))
		inputs_mask = jnp.arange(I) < rpn_degree
		inputs = jnp.where(inputs_mask, inputs, 0)
		outputs = self._evaluate_expr(rpn_expr, rpn_coeffs, rpn_degree, inputs, O)

		if self.opts.poly.int_base is None:
			inputs_enc = inputs + self.zero_token
			outputs_enc = outputs + self.zero_token
			input_logical_sz, output_logical_sz = I, O
			input_sz, output_sz = I, O
			outputs_places = jnp.arange(O) 
		else:
			def last_found_index(ary, val):
				return jnp.max(jnp.where(ary == val, jnp.arange(ary.shape[0]), -1)) 
			tokenize_opts = (
				self.opts.poly.int_base, self.opts.use_dpse, self.zero_token, self.plus_token,
				self.minus_token, self.pad_token)
			outputs_mask = jnp.full_like(outputs, True)
			inputs_enc, inputs_places = jfuncs.tokenize_ints(inputs, inputs_mask, *tokenize_opts)
			outputs_enc, outputs_places = jfuncs.tokenize_ints(outputs, outputs_mask, *tokenize_opts)
			input_logical_sz = last_found_index(inputs_places, rpn_degree - 1) + 1
			output_logical_sz = last_found_index(outputs_places, O - 1) + 1
			input_sz = inputs_enc.shape[0]
			output_sz = outputs_enc.shape[0]

		obs_sym = jnp.full((R + 1 + input_sz + output_sz,), self.pad_token, dtype=jnp.int32)

		match self.opts.task_ty:
			case TaskType.PROGRAM_EXECUTION: 
				"""
				| [RPN_EXPR]  | [=]  | [INPUT] | [OUTPUT] |
				r_beg         e_beg  i_beg       o_beg    sym_end
				"""
				r_beg = 0
				e_beg = self.rpn_sizes[e]
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
				sym_end = r_beg + self.rpn_sizes[e]
				pred_beg = r_beg
			case _:
				raise RuntimeError(f"Unrecognized task type: {self.opts.task_ty}")

		obs_sym = jfuncs.copy_range(obs_sym, self.rpn_tokens[e], r_beg, 0, self.rpn_sizes[e])
		obs_sym = obs_sym.at[e_beg].set(self.equals_token)
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
				out_code = jnp.where(jnp.arange(R) > self.rpn_sizes[e], -1, out_code)
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
				self.equals_token,
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
				return np.array([self.print_expr(t) for t in self.rpn_tokens])
			case _:
				raise RuntimeError(f"Unrecognized cat: {cat}")
