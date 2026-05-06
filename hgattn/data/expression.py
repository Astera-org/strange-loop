import itertools
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

@dataclass(frozen=True)
class InductiveOpts:
	int_base: int|None   # base for encoding large integers (if None, do not base encode
	use_dpse: bool       # use digit-position specific embeddings
	max_expr_depth: int  # maximum depth for generating expressions
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
	
	def __post_init__(self):
		try:
			object.__setattr__(self, 'binops', tuple(BinaryOp(op) for op in self.binops))
		except ValueError as v:
			raise ValueError(
					f"Received one or more invalid binops in {self.binops}.  "
					f"Valid binops are {', '.join(op.value for op in BinaryOp)}") from v

		try:
			object.__setattr__(self, 'uops', tuple(UnaryOp(op) for op in self.uops))
		except ValueError as v:
			raise ValueError(
					f"Received one or more invalid uops in {self.uops}.  "
					f"Valid uops are {', '.join(op.value for op in UnaryOp)}") from v

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

	def __init__(self, opts: InductiveOpts, is_train: bool, seed: int):
		# jax.config.update("jax_enable_x64", True)
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

		rng = np.random.default_rng(seed=seed)

		tg = arith.TreeGen(opts.binops, opts.uops, variables, all_const_vals)

		all_trees = []
		for nvars in opts.allowed_num_vars:
			for nconsts in opts.allowed_num_consts:
				trees = tg.gen_trees(seed, nvars, nconsts, opts.max_expr_depth, opts.n_exprs * 2) 
				all_trees.extend(trees)

		rpns = tuple(arith.RPNExpression(t, self.opts.mod_val) for t in all_trees)
		rpns = rpns[:opts.n_exprs]
		print(f"found {len(rpns)} rpns after filtering")
		# print("\n".join(rpn.infix() for rpn in rpns))

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

		self.rpn_exprs = ragged_stack(rpn_exprs, switch_code_map['NOOP'])
		self.rpn_tokens = ragged_stack(rpn_toks, self.pad_token)

		# rpn_tokens is encoded in token_map alphabet.  it stores const values in
		# base encoded format if int_base is not None, or plain format if None
		self.rpn_consts = ragged_stack([np.array(rpn.const_values) for rpn in rpns], self.pad_token)
		self.rpn_degree = jnp.array([get_degree(rpn.variables) for rpn in rpns])

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

	def _generate_one(self, key):
		expr_key, input_key = jax.random.split(key)
		e = jax.random.choice(expr_key, self.opts.n_exprs)
		rpn_expr = self.rpn_exprs[e,:]
		rpn_consts = self.rpn_consts[e,:]
		rpn_degree = self.rpn_degree[e]
		tokens = self.rpn_tokens[e,:]
		I, O = self.opts.n_vars, self.opts.n_outputs

		rpn_token_string = jnp.concatenate((tokens, jnp.array(self.equals_token)[None]))

		R = rpn_token_string.shape[0]

		inputs = jax.random.choice(
				input_key, 
				jnp.arange(self.opts.input_beg, self.opts.input_end),
				(I,))

		evaluate_fn = partial(evaluate_rpn, self.opts.max_expr_depth, self.opts.mod_val)

		def step_fn(state, _):
			variables = state
			next_var = evaluate_fn(rpn_expr, rpn_consts, variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

		_, output = jax.lax.scan(step_fn, inputs, length=O)

		rpn_pad_mask = rpn_token_string != self.pad_token
		inputs_pad_mask = jnp.arange(I) > I - rpn_degree - 1
		output_pad_mask = jnp.full((O,), True)

		input_hash = jfuncs.hash(jnp.where(inputs_pad_mask, inputs, 0))

		series = jnp.concatenate((inputs, output))
		S = series.shape[0]
		series_pad_mask = jnp.concatenate((inputs_pad_mask, output_pad_mask))

		if self.opts.int_base is None:
			series_enc, ntoks = jfuncs.compact_masked(series + self.zero_token, series_pad_mask)
			series_enc = jnp.where(jnp.arange(S) < ntoks, series_enc, self.pad_token)
			series_positions = jfuncs.masked_arange(series_pad_mask) 

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
		jax.debug.print(
			"series_enc: {}\nseries: {}\ninputs: {}\noutput: {}\n"
			"inputs_pad_mask: {}\noutputs_pad_mask: {}\n", 
			series_enc, 
			series,
			inputs,
			output,
			inputs_pad_mask,
			output_pad_mask)
		"""
		
		series_enc_pad_mask = series_positions > -1
		obs_sym_pad_mask = jnp.concatenate((rpn_pad_mask, series_enc_pad_mask))
		target_pad_mask = jnp.concatenate((jnp.full((R,), False), series_positions >= rpn_degree))
		obs_sym = jnp.concatenate((rpn_token_string, series_enc))
		obs_sym, obs_sym_ntok = jfuncs.compact_masked(obs_sym, obs_sym_pad_mask)
		input_mask = jnp.arange(obs_sym.shape[0]) < obs_sym_ntok
		target_mask, _ = jfuncs.compact_masked(target_pad_mask, obs_sym_pad_mask)

		return obs_sym, input_mask, target_mask, input_hash

	def _gen_one_item(self, key: PRNGKeyArray) -> TokensAndProbs:
		obs_sym_C, input_mask_C, target_mask_C, input_hash = self._generate_one(key)
		obs_prob_C = jax.nn.one_hot(obs_sym_C, self.vocab_size)
		is_train_frac = (input_hash % 1024) < int(self.opts.train_frac * 1024)
		is_active = (self.is_train == is_train_frac)

		return TokensAndProbs(
				key=jax.random.key_data(key), 
				obs_sym=obs_sym_C,
				obs_prob=obs_prob_C,
				input_mask=input_mask_C,
				target_mask=target_mask_C,
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

		# jax.debug.print("before: obs_sym:\n{}\nactive:\n{}\n", item.obs_sym[:,10], item.active)

		item = jax.tree.map(_fraction, item)

		# jax.debug.print( "after: item.obs_sym:\n{}\nactive:\n{}\n", item.obs_sym[:,10], item.active)

		return item

