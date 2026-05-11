import itertools
import math
import random
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import operator
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Union, Iterable, Any
from jaxtyping import PRNGKeyArray, Array
from .. import jfuncs
from ..tools import arith
from ..tools.arith import BinaryOp, UnaryOp, ControlOp
from .types import TokensAndProbs

def get_variables(n: int):
	return tuple(reversed(tuple(chr(ord('A') + i) for i in range(n))))

def get_constants(n: int):
	return tuple('c' + chr(ord('A') + i) for i in range(n))

def get_degree(rpn_vars: list[str]) -> int:
	if len(rpn_vars) == 0:
		return 0
	return ord(max(rpn_vars)) - ord('A') + 1

def get_binds(rpn_vars: list[str], values: list[int]) -> dict[str, int]:
	return { var: values[-(ord(var) - ord('A') + 1)] for var in rpn_vars }

# Used in rpn_step
def get_switch_codes(n_vars: int, n_consts: int):
	_vars = get_variables(n_vars)
	_consts = get_constants(n_consts)
	return ('NOOP',) + tuple(BinaryOp) + tuple(UnaryOp) + _vars + _consts

def rpn_step(global_mod_val: int, state, rpn_token):
	"""
	Step for scanning an RPN expression.  global_mod_val is only necessary
	if the expressions will use MOD_* functions (see arith.py)
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

	def safe_floor_divide(x, y):
		return jnp.where(y == 0, 0, jnp.floor_divide(x, y))

	# the order here must correspond with PAD + all_ops_strings below (hack) 
	branches = [
		no_op,
		lambda: binary(jnp.add),
		lambda: binary(jnp.subtract),
		lambda: binary(jnp.multiply),
		lambda: binary(safe_floor_divide),
		lambda: binary(jnp.mod),
		lambda: binary(mod_fn(jnp.add)),
		lambda: binary(mod_fn(jnp.subtract)),
		lambda: binary(mod_fn(jnp.multiply)),
		lambda: binary(mod_fn(safe_floor_divide)),
		lambda: unary(jnp.abs),
		lambda: unary(lambda x: jnp.power(x, 2)),
		lambda: unary(lambda x: jnp.where(x >= 0, 1, -1)),
		lambda: unary(lambda x: jnp.maximum(0, x)),
		lambda: unary(mod_fn(lambda x: jnp.power(x, 2))),
	]

	for i in range(variables.shape[0]):
		branches.append(lambda i=i: push(variables[i]))
	
	for i in range(constants.shape[0]):
		branches.append(lambda i=i: push(constants[i]))

	new_stack, new_ptr = jax.lax.switch(rpn_token, branches)
	return (new_stack, new_ptr, constants, variables), None

def evaluate_rpn(
	max_expr_depth: int, 
	global_mod_val: int,
	rpn_expr: Array, 
	rpn_consts: Array, 
	variables: Array,
):
	stack = jnp.empty((max_expr_depth * 2,), dtype=jnp.int32)
	ptr = jnp.array(0, dtype=jnp.int32)
	state = stack, ptr, rpn_consts, variables
	step_fn = partial(rpn_step, global_mod_val)
	final_state, _ = jax.lax.scan(step_fn, state, rpn_expr)
	final_stack = final_state[0]
	ans = final_stack[0]
	return ans

class SplitType(Enum):
	INPUT = "input"
	EXPR = "expr"

class TargetCategory(Enum):
	CTX_POS = "ctx_pos"
	EXPR = "expr"

@dataclass
class InductiveOpts:
	int_base: int|None   # base for encoding large integers (if None, do not base encode
	use_dpse: bool       # use digit-position specific embeddings
	max_expr_depth: int  # maximum depth for generating expressions
	min_entropy_frac: float # min fraction of maximal entropy in [0, 1) to retain an rpn
	allowed_num_consts: list[int] # filter by number of consts appearing
	allowed_num_vars: list[int]   # filter by number of variables appearing
	n_exprs: int
	binops: tuple[BinaryOp]    # which binary ops to use (legal values of BinaryOp)
	uops: tuple[UnaryOp]      # which unary ops to use (legal values of UnaryOp)
	const_beg: int       # minimum value of a constant (must be >= 0)
	const_end: int       # limit (exclusive value of a constant (must be >= 1)
	input_beg: int       
	input_end: int       # sample inputs in [input_beg, input_end)
	n_vars: int          # number of different variables that can appear
	n_consts: int        # number of different constants that can appear
	n_outputs: int       # number of output values to generate using the formula
	mod_val: int|None    # used for MOD_* ops in arith.{BinaryOp,UnaryOp}
	train_frac: float    # fraction in [0, 1] for training split
	split_ty: SplitType # strategy for train/test split
	
	def __post_init__(self):
		try:
			self.binops = tuple(BinaryOp(op) for op in self.binops)
		except ValueError as v:
			raise ValueError(
					f"Received one or more invalid binops in {self.binops}.  "
					f"Valid binops are {', '.join(op.value for op in BinaryOp)}") from v

		try:
			self.uops = tuple(UnaryOp(op) for op in self.uops)
		except ValueError as v:
			raise ValueError(
					f"Received one or more invalid uops in {self.uops}.  "
					f"Valid uops are {', '.join(op.value for op in UnaryOp)}") from v

		try:
			self.split_ty = SplitType(self.split_ty)
		except ValueError as v:
			raise ValueError(
					f"Received split_ty {self.split_ty}.  "
					f"Valid split_ty are {', '.join(s.value for s in SplitType)}") from v

		if self.n_consts > len(range(self.const_beg, self.const_end)):
			raise RuntimeError(
				f"Received {self.n_consts=}, which exceeds the range of allowable "
				f"constants [{self.const_beg}, {self.const_end})")
		if self.n_vars > 10:
			raise RuntimeError(f"Received {self.n_vars=} exceeding max of 10")
		if self.mod_val is not None and self.input_end > self.mod_val:
			raise RuntimeError(f"Received {self.input_end=} exceeding {self.mod_val=}")
		if self.mod_val is not None and self.const_end > self.mod_val:
			raise RuntimeError(f"Received {self.const_end=} exceeding {self.mod_val=}")

class InductiveDataset(eqx.Module):
	opts: InductiveOpts = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	pad_token: int = eqx.field(static=True)
	plus_token: int = eqx.field(static=True)
	minus_token: int = eqx.field(static=True)
	equals_token: int = eqx.field(static=True)
	zero_token: int = eqx.field(static=True)
	num_digit_tokens: int = eqx.field(static=True)
	inv_token_map: dict = eqx.field(static=True)
	is_train: bool = eqx.field(static=True)
	rpn_exprs: jax.Array
	rpn_tokens: jax.Array
	rpn_consts: jax.Array
	rpn_degree: jax.Array
	rpn_sizes: jax.Array

	def __init__(self, opts: InductiveOpts, is_train: bool, seed: int):
		# jax.config.update("jax_enable_x64", True)
		key = jax.random.key(seed)

		self.opts = opts
		self.is_train = is_train
		used_ops = opts.binops + opts.uops
		all_const_vals = list(range(opts.const_beg, opts.const_end))
		switch_codes = get_switch_codes(opts.n_vars, opts.n_consts)
		variables = get_variables(opts.n_vars)
		const_names = get_constants(opts.n_consts)

		if opts.int_base is None:
			controls = tuple(ControlOp) 
			self.num_digit_tokens = opts.mod_val
		else:
			max_val = 2**63 if opts.mod_val is None else opts.mod_val
			D = jfuncs.get_max_digits(max_val, opts.int_base)
			if opts.use_dpse:
				self.num_digit_tokens = D * opts.int_base
			else:
				self.num_digit_tokens = opts.int_base
			controls = tuple(ControlOp)

		sym_tokens = ('PAD',) + used_ops + controls + variables
		token_map = { tok: idx for idx, tok in enumerate(sym_tokens) }
		self.vocab_size = len(token_map) + self.num_digit_tokens
		self.zero_token = max(token_map.values()) + 1

		self.pad_token = token_map['PAD']
		self.plus_token = token_map[ControlOp.PLUS]
		self.minus_token = token_map[ControlOp.MINUS]
		self.equals_token = token_map[ControlOp.EQUALS]

		switch_code_map = { tok: idx for idx, tok in enumerate(switch_codes) }

		self.inv_token_map = { v: k for k, v in token_map.items() }

		tg = arith.TreeGen(opts.binops, opts.uops, variables, all_const_vals)

		all_trees = []
		for nvars in opts.allowed_num_vars:
			for nconsts in opts.allowed_num_consts:
				trees = tg.gen_trees(seed, nvars, nconsts, opts.max_expr_depth, opts.n_exprs * 2) 
				all_trees.extend(trees)

		rpns = tuple(arith.RPNExpression(t, self.opts.mod_val) for t in all_trees)
		rpns = rpns[:opts.n_exprs]
		print(f"found {len(rpns)} rpns")

		rpn_exprs = [self.to_rpn_tokens(switch_code_map, r, const_names) for r in rpns]

		if opts.int_base is None:
			rpn_toks = [r.tokens(token_map, self.zero_token) for r in rpns]
		else:
			rpn_toks = [r.tokens_base_enc(
				token_map, opts.use_dpse, opts.int_base, self.zero_token)
			   for r in rpns
			]

		def ragged_stack(arrays, pad):
			N = len(arrays)
			D = max((len(a) for a in arrays), default=0)
			result = np.full((N, D), pad, dtype=arrays[0].dtype)
			for idx, ary in enumerate(arrays):
				result[idx,:len(ary)] = ary
			return jnp.array(result)

		rpn_exprs = ragged_stack(rpn_exprs, switch_code_map['NOOP'])
		rpn_tokens = ragged_stack(rpn_toks, self.pad_token)
		rpn_consts = ragged_stack(
			[np.array(rpn.const_values, dtype=np.int64) for rpn in rpns], 
			self.pad_token)
		rpn_degree = jnp.array([get_degree(rpn.variables) for rpn in rpns])
		rpn_sizes = jnp.array([t.shape[0] for t in rpn_toks])

		key_E = jax.random.split(key, num=rpn_exprs.shape[0])
		ent_fn = lambda xs: self._expr_entropy_fraction(*xs, 1000)
		ent_frac_E = jax.lax.map(ent_fn, (key_E, rpn_exprs, rpn_consts, rpn_degree), batch_size=1024)
		active_expr_E = ent_frac_E > self.opts.min_entropy_frac

		rpn_exprs, n_active = jfuncs.compact_masked(rpn_exprs, active_expr_E)
		rpn_tokens, _ = jfuncs.compact_masked(rpn_tokens, active_expr_E)
		rpn_consts, _ = jfuncs.compact_masked(rpn_consts, active_expr_E)
		rpn_degree, _ = jfuncs.compact_masked(rpn_degree, active_expr_E)
		rpn_sizes, _ = jfuncs.compact_masked(rpn_sizes, active_expr_E)

		"""
		jax.debug.print(
			"active_expr_E:\n{}\n"
			"ent_frac_E:\n{}\n",
			active_expr_E, ent_frac_E
		)
		"""
		print(f"Found {n_active} rpns with > {self.opts.min_entropy_frac} entropy fraction")

		self.rpn_exprs = rpn_exprs[:n_active]
		self.rpn_tokens = rpn_tokens[:n_active]
		self.rpn_consts = rpn_consts[:n_active]
		self.rpn_degree = rpn_degree[:n_active]
		self.rpn_sizes = rpn_sizes[:n_active]

	@property
	def num_expressions(self):
		return self.rpn_exprs.shape[0]

	@eqx.filter_jit
	def _expr_entropy_fraction(
		self,
		key: PRNGKeyArray,
		rpn_expr: jax.Array,
		rpn_consts: jax.Array,
		rpn_degree: jax.Array,
		num_trials: int
	) -> jax.Array:
		"""
		Compute average entropy fraction for the `rpn_expr` (plugging in `rpn_consts`
		during the eval).  Evaluate `num_trials` to compute the average.
		"""
		B, I, O = num_trials, self.opts.n_vars, self.opts.n_outputs

		inputs_BI = jax.random.choice(
			key, jnp.arange(self.opts.input_beg, self.opts.input_end), (B, I))

		eval_fn = jax.vmap(self._evaluate_expr, in_axes=(None, None, None, 0, None))

		outputs_BC = eval_fn(rpn_expr, rpn_consts, rpn_degree, inputs_BI, O)

		# infer max bins 
		all_modulo_ops = all(op in arith.MODULO_OPS for op in self.opts.binops + self.opts.uops)
		if all_modulo_ops and self.opts.mod_val is not None:
			max_bins = self.opts.mod_val
		else:
			max_bins = B * O 

		return arith.entropy_fraction(outputs_BC, max_bins)

	def to_rpn_tokens(
		self, 
		switch_code_map: dict[Any, int],
		rpn: arith.RPNExpression, 
		const_names: list[str],
	):
		"""
		Convert RPN expression to a string in the rpn_token_map alphabet
		"""
		toks = []
		for v in rpn.token_vals:
			match v:
				case int(const_val):
					ci = rpn.const_values.index(const_val)
					toks.append(switch_code_map[const_names[ci]])
				case str(s):
					toks.append(switch_code_map[s])
				case BinaryOp() | UnaryOp():
					toks.append(switch_code_map[v])
				case _:
					raise RuntimeError(f"Unrecognized RPN token val: {v}")
		return np.array(toks)

	def token_to_digit_value(self, token: int) -> int:
		val = token - self.zero_token
		if self.opts.int_base is None:
			return val
		return val % self.opts.int_base

	def _decode_tokens_enc(self, tokens: np.array) -> list[arith.RPNValue]:
		"""
		Decodes tokens (the 'obs_sym' field), which contains base-token encoded
		integers and string representations of RPNValue.
		"""
		sign, curval, place = None, None, None
		results = []
		plus = self.plus_token
		minus = self.minus_token
		zero = self.zero_token

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
					_, val = divmod(val, self.opts.int_base)
				curval = val * place + curval
				place *= self.opts.int_base
			else:
				if curval is not None:
					results.append(sign * curval)
					curval = None
				sym = self.inv_token_map.get(tok)
				if sym is None:
					import pdb
					pdb.set_trace()
					raise RuntimeError(f"Could not find symbol for token {tok}")
				results.append(sym)

		if curval is not None:
			results.append(sign * curval)

		return results

	def _decode_tokens_no_enc(self, tokens: np.array) -> list[arith.RPNValue]:
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

	def decode_tokens(self, tokens: np.array) -> list[arith.RPNValue]:
		if self.opts.int_base is None:
			return self._decode_tokens_no_enc(tokens)
		return self._decode_tokens_enc(tokens)

	def _split(self, tokens: np.array) -> tuple:
		inds, = np.nonzero(tokens == self.equals_token)
		if inds.shape[0] != 1:
			import pdb
			pdb.set_trace()
			raise RuntimeError(
				"Symbol string must have exactly one 'EQUALS' token.  "
				f"Has {inds.shape[0]}")
		return tokens[:inds[0]], tokens[inds[0]+1:]

	def _trim(self, vals: list[arith.RPNValue]) -> list[arith.RPNValue]:
		try:
			end = vals.index('PAD')
		except ValueError:
			end = len(vals)
		return vals[:end]

	def _split_and_trim(self, tokens: np.array) -> tuple:
		# return the rpn_vals, series_vals pair
		rpn_tokens, series_tokens = self._split(tokens)
		rpn_vals = self.decode_tokens(rpn_tokens)
		series_vals = self.decode_tokens(series_tokens)
		series_vals = self._trim(series_vals)
		return rpn_vals, series_vals

	def print(self, tokens: np.array) -> str:
		rpn_vals, series_vals = self._split_and_trim(tokens)
		rpn = arith.RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
		return rpn.infix() + " = " + " ".join(str(s) for s in series_vals)

	def print_raw(self, tokens: np.array) -> str:
		rpn_tokens, series_tokens = self._split(tokens)
		rpn_vals = self.decode_tokens(rpn_tokens)
		rpn = arith.RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
		res = []
		for tok in series_tokens.tolist():
			obj = self.inv_token_map.get(tok)
			match obj:
				case None:
					val = tok - self.zero_token
					if self.opts.int_base and self.opts.use_dpse:
						place, val = divmod(val, self.opts.int_base)
						s = f"p{place}_{val}"
					else:
						s = str(val)
				case str(s):
					pass
				case _: 
					s = obj.value
			res.append(s)

		res = [ '+' if s == 'plus_sign' else s for s in res ]
		res = self._trim(res)
		return rpn.infix() + " = " + " ".join(res)

	def validate(self, tokens: np.array) -> tuple[bool, str]:
		"""
		parse obs_sym tokens into the expression and integer series
		"""
		rpn_vals, series_vals = self._split_and_trim(tokens)

		try:
			rpn = arith.RPNExpression.from_vals(rpn_vals, self.opts.mod_val)
		except RuntimeError as ex:
			import pdb
			pdb.set_trace()

		degree = get_degree(rpn.variables)
		if len(series_vals) < degree:
			return False, f"Got degree {degree} RPN but {series_vals.shape[0]} series values"

		for i in range(degree, len(series_vals)):
			binds = get_binds(rpn.variables, series_vals[i-degree:i])
			ans = rpn.evaluate(**binds)
			if ans != series_vals[i]:
				return False, (
					f"{ans=} != series_vals[{i}]={series_vals[i]}, "
					f"{binds=}\n"
					f"{rpn=}\n"
					f"{series_vals=}\n")
		return True, (
			f"{rpn=}\n"
			f"{rpn_vals=}\n"
			f"{series_vals=}\n"
		)

	def _evaluate_expr(
		self,
		rpn_expr: jax.Array,
		rpn_consts: jax.Array,
		rpn_degree: jax.Array,
		inputs: jax.Array,
		num_outputs: int
	) -> jax.Array:
		"""
		Evaluate `rpn_expr` `num_outputs` times, plugging in `rpn_consts` and
		`inputs`.

		"""
		evaluate_fn = partial(evaluate_rpn, self.opts.max_expr_depth, self.opts.mod_val)

		def step_fn(state, _):
			variables = state
			next_var = evaluate_fn(rpn_expr, rpn_consts, variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

		init_state = jnp.roll(inputs, -rpn_degree)
		_, output = jax.lax.scan(step_fn, init_state, length=num_outputs)
		return output

	def _generate_one(self, key):
		expr_key, input_key = jax.random.split(key)

		I, O = self.opts.n_vars, self.opts.n_outputs
		E, R = self.rpn_exprs.shape
		C = R + 1 + I + O
		obs_sym = jnp.full((C,), self.pad_token, dtype=jnp.int32)

		e = jax.random.choice(expr_key, E)
		rpn_expr = self.rpn_exprs[e]
		rpn_consts = self.rpn_consts[e]
		rpn_degree = self.rpn_degree[e]
		rpn_end = self.rpn_sizes[e]

		input_rng = jnp.arange(self.opts.input_beg, self.opts.input_end)
		inputs = jax.random.choice(input_key, input_rng, (I,))
		inputs = jnp.where(jnp.arange(I) < rpn_degree, inputs, 0)
		outputs = self._evaluate_expr(rpn_expr, rpn_consts, rpn_degree, inputs, O)

		input_toks = inputs + self.zero_token
		output_toks = outputs + self.zero_token

		r_beg = 0
		e_beg = self.rpn_sizes[e]
		i_beg = e_beg + 1
		o_beg = i_beg + rpn_degree
		sym_end = o_beg + O

		obs_sym = jax.lax.dynamic_update_slice(obs_sym, self.rpn_tokens[e], (r_beg,))
		obs_sym = obs_sym.at[e_beg].set(self.equals_token)
		obs_sym = jax.lax.dynamic_update_slice(obs_sym, input_toks, (i_beg,))
		obs_sym = jax.lax.dynamic_update_slice(obs_sym, output_toks, (o_beg,))

		inp_mask = jnp.arange(C) < sym_end

		Obits = math.ceil(math.log2(O))
		out_cats = (e << Obits)[None] + jax.lax.iota(jnp.int32, O) 
		target_cat = jnp.full((C,), -1, dtype=jnp.int32)
		target_cat = jax.lax.dynamic_update_slice(target_cat, out_cats, (o_beg,))

		match self.opts.split_ty:
			case SplitType.INPUT:
				split_hash = jfuncs.hash(inputs)
			case SplitType.EXPR:
				split_hash = jfuncs.hash(e)
			case _:
				raise RuntimeError(f"Unrecognized split type: {self.opts.split_ty.value}")

		"""
		else:
			series_enc, series_positions = jfuncs.tokenize_ints(
				series,
				series_pad_mask,
				self.opts.int_base,
				self.opts.use_dpse,
				self.zero_token,
				self.plus_token,
				self.minus_token,
				self.pad_token
			)
		"""
		return obs_sym, inp_mask, trg_mask, target_cat, split_hash 

	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		obs_sym_C, input_mask_C, target_cat_C, split_hash = self._generate_one(key)
		obs_prob_C = jax.nn.one_hot(obs_sym_C, self.vocab_size)
		is_train_frac = (split_hash % 1024) < int(self.opts.train_frac * 1024)
		is_active = (self.is_train == is_train_frac)

		return TokensAndProbs(
				key=jax.random.key_data(key), 
				obs_sym=obs_sym_C,
				obs_prob=obs_prob_C,
				input_mask=input_mask_C,
				target_cat=target_cat_C,
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

	def get_target_cat(
		self,
		mode: TargetCategory, 
		target_cat: jax.Array,
	) -> jax.Array:
		match mode:
			case TargetCategory.CTX_POS:
				obits = (jnp.uint32(1) << self.opts.n_outputs) - 1
				ctx_vals = jnp.bitwise_and(obits, target_cat)
				return jnp.where(target_cat == -1, -1, ctx_vals)
			case TargetCategory.EXPR:
				expr_vals = target_cat >> self.opts.n_outputs
				return jnp.where(target_cat == -1, -1, expr_vals)
			case _:
				raise RuntimeError(f"Unrecognized mode: {mode}")

	def get_target_init(
		self,
		mode: TargetCategory, 
	) -> jax.Array:
		match mode:
			case TargetCategory.CTX_POS:
				return jnp.zeros((self.opts.n_outputs,))
			case TargetCategory.EXPR:
				return jnp.zeros((self.rpn_exprs.shape[0],))
			case _:
				raise RuntimeError(f"Unrecognized mode: {mode}")


