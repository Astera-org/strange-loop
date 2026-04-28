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
from . import arith
from .arith import BinaryOp, UnaryOp
from .types import TokensAndProbs

def get_variables(n: int):
	return tuple(reversed(tuple(chr(ord('A') + i) for i in range(n))))

def get_constants(n: int):
	return tuple('c' + chr(ord('A') + i) for i in range(n))

def get_degree(rpn_vars: list[str]) -> int:
	return ord(max(rpn_vars)) - ord('A') + 1

def get_binds(rpn_vars: list[str], values: list[int]) -> dict[str, int]:
	return { var: values[-(ord(var) - ord('A') + 1)] for var in rpn_vars }

def rpn_step(state, rpn_token):
	"""
	Step for scanning an RPN expression
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

	branches = [
		no_op,
		lambda: binary(jnp.add),
		lambda: binary(jnp.subtract),
		lambda: binary(jnp.multiply),
		lambda: binary(jnp.floor_divide),
		lambda: binary(jnp.mod),
		lambda: unary(jnp.abs),
		lambda: unary(lambda x: jnp.power(x, 2)),
		lambda: unary(lambda x: jnp.where(x >= 0, 1, -1)),
		lambda: unary(lambda x: jnp.maximum(0, x))
	]

	for i in range(variables.shape[0]):
		branches.append(lambda i=i: push(variables[i]))
	
	for i in range(constants.shape[0]):
		branches.append(lambda i=i: push(constants[i]))

	new_stack, new_ptr = jax.lax.switch(rpn_token, branches)
	return (new_stack, new_ptr, constants, variables), None

def evaluate_rpn(
	max_expr_depth: int, 
	rpn_expr: Array, 
	rpn_consts: Array, 
	variables: Array,
):
	stack = jnp.empty((max_expr_depth * 2,), dtype=jnp.int64)
	ptr = jnp.array(0, dtype=jnp.int32)
	state = stack, ptr, rpn_consts, variables
	final_state, _ = jax.lax.scan(rpn_step, state, rpn_expr)
	final_stack = final_state[0]
	ans = final_stack[0]
	return ans

@dataclass(frozen=True)
class InductiveOpts:
	context_len: int     # [expression] [inputs] [outputs] [padding]
	int_base: int        # base for encoding large integers
	max_expr_depth: int  # maximum depth for generating expressions
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
	series_mod_val: int|None # wrap each recurrence in a (val) mod `series_mod_val`
	
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

class InductiveDataset(eqx.Module):
	opts: InductiveOpts = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	rpn_token_map: dict = eqx.field(static=True)
	token_map: dict = eqx.field(static=True)
	inv_token_map: dict = eqx.field(static=True)
	rpn_exprs: jax.Array
	rpn_tokens: jax.Array
	rpn_consts: jax.Array
	rpn_degree: jax.Array

	def __init__(self, opts: InductiveOpts, is_train: bool, seed: int):
		jax.config.update("jax_enable_x64", True)
		self.opts = opts
		all_ops = tuple(BinaryOp) + tuple(UnaryOp)
		all_ops_strings = tuple(op.value for op in BinaryOp) + tuple(op.value for op in UnaryOp)
		all_const_vals = list(range(opts.const_beg, opts.const_end))
		variables = get_variables(opts.n_vars)
		const_names = get_constants(opts.n_consts)

		digits = tuple(str(d) for d in range(opts.int_base))
		controls = 'PLUS_SIGN', 'MINUS_SIGN', 'EQUALS'
		rpn_tokens = ('PAD',) + all_ops_strings + variables + const_names 
		sym_tokens = ('PAD',) + all_ops_strings + variables + controls + digits
		sym_tokens_obj = ('PAD',) + all_ops + variables + controls + digits

		self.rpn_token_map = { tok: idx for idx, tok in enumerate(rpn_tokens) }
		self.token_map = { tok: idx for idx, tok in enumerate(sym_tokens) }
		self.inv_token_map = { idx: tok for idx, tok in enumerate(sym_tokens_obj) }

		self.vocab_size = len(self.token_map)
		rng = np.random.default_rng(seed=seed)

		nodes = list(
				arith.gen_expressions(
					rng,
					opts.max_expr_depth,
					opts.binops,
					opts.uops,
					opts.n_consts,
					opts.n_vars,
					all_const_vals,
					variables)
				)
		if len(nodes) < opts.n_exprs:
			raise RuntimeError(
					f"Generated only {len(exprs)} expressions but require {opts.n_exprs}")

		def ragged_stack(arrays, pad):
			N = len(arrays)
			D = max(len(a) for a in arrays)
			result = np.full((N, D), pad, dtype=arrays[0].dtype)
			for idx, ary in enumerate(arrays):
				result[idx,:len(ary)] = ary
			return jnp.array(result)

		rpns = tuple(arith.RPNExpression(n) for n in nodes)
		rpns = tuple(rpn for rpn in rpns if len(rpn.variables) > 0)
		inds = rng.choice(np.arange(len(rpns)), opts.n_exprs, replace=False)
		rpns = tuple(rpns[i] for i in inds)

		PAD = self.rpn_token_map['PAD']

		
		# rpn_exprs is encoded in the rpn_token_map.  it stores const names
		self.rpn_exprs = ragged_stack([self.to_rpn_tokens(rpn, const_names) for rpn in rpns], PAD)

		# rpn_tokens is encoded in token_map alphabet.  it stores const values in
		# base encoded format
		self.rpn_tokens = ragged_stack([self.to_tokens(rpn) for rpn in rpns], PAD)
		self.rpn_consts = ragged_stack([np.array(rpn.const_values) for rpn in rpns], PAD)
		self.rpn_degree = jnp.array([get_degree(rpn.variables) for rpn in rpns])

	def to_rpn_tokens(self, rpn: arith.RPNExpression, const_names):
		"""
		Convert RPN expression to a string in the rpn_token_map alphabet
		"""
		toks = []
		for v in rpn.token_vals:
			match v:
				case int(const_val):
					ci = rpn.const_values.index(const_val)
					toks.append(self.rpn_token_map[const_names[ci]])
				case str(s):
					toks.append(self.rpn_token_map[s])
				case BinaryOp() | UnaryOp():
					toks.append(self.rpn_token_map[v.value])
				case _:
					raise RuntimeError(f"Unrecognized RPN token val: {v}")
		return np.array(toks)

	def to_tokens(self, rpn: arith.RPNExpression):
		"""
		Convert RPN expression to a string in the self.token_map alphabet
		"""
		toks = []
		for v in rpn.token_vals:
			match v:
				case BinaryOp() | UnaryOp():
					toks.append(self.token_map[v.value])
				case int(const_val): 
					enc = jfuncs.tokenize_one_int(
							const_val, 
							self.opts.int_base, 
							self.token_map['0'],
							self.token_map['PLUS_SIGN'],
							self.token_map['MINUS_SIGN'])
					toks.extend(enc.tolist())
				case str(s):
					toks.append(self.token_map[s])
				case _:
					raise RuntimeError(f"Unrecognized RPN token val: {v}")
		# print(rpn.token_vals, toks)
		return np.array(toks) 

	def decode_tokens(self, tokens: np.array) -> list[arith.RPNValue]:
		"""
		Decodes tokens (the 'obs_sym' field), which contains base-token encoded
		integers and string representations of RPNValue.
		"""
		sign, curval = None, None
		results = []
		plus = self.token_map['PLUS_SIGN']
		minus = self.token_map['MINUS_SIGN']
		zero = self.token_map['0']
		digits = range(zero, zero + self.opts.int_base)

		for tok in tokens.tolist():
			if tok in (plus, minus):
				if curval is not None:
					results.append(sign * curval)
				sign = 1 if tok == plus else -1
				curval = 0
			elif tok in digits:
				if curval is None:
					raise RuntimeError(f"Invalid symbol sequence")
				val = tok - zero
				curval = curval * self.opts.int_base + val
			else:
				if curval is not None:
					results.append(sign * curval)
					curval = None
				sym = self.inv_token_map[tok]
				results.append(sym)

		if curval is not None:
			results.append(sign * curval)

		return results

	def validate(self, tokens: np.array) -> tuple[bool, str]:
		"""
		parse obs_sym tokens into the expression and integer series
		"""
		inds, = np.nonzero(tokens == self.token_map['EQUALS'])
		if inds.shape[0] != 1:
			return False, (
					"Symbol string must have exactly one 'EQUALS' token.  "
					f"Has {inds.shape[0]}")
		rpn_vals = self.decode_tokens(tokens[:inds[0]])
		series_vals = self.decode_tokens(tokens[inds[0]+1:])
		try:
			end = series_vals.index('PAD')
		except ValueError:
			end = len(series_vals)
		series_vals = series_vals[:end]

		rpn = arith.RPNExpression.from_vals(rpn_vals)
		degree = get_degree(rpn.variables)
		if len(series_vals) < degree:
			return False, f"Got degree {degree} RPN but {series_vals.shape[0]} series values"

		for i in range(degree, end):
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

		rpn_token_string = jnp.concatenate((tokens, jnp.array(self.token_map['EQUALS'])[None]))
		R = rpn_token_string.shape[0]

		inputs = jax.random.choice(
				input_key, 
				jnp.arange(self.opts.input_beg, self.opts.input_end),
				(I,))

		evaluate_fn = partial(evaluate_rpn, self.opts.max_expr_depth)

		def step_fn(state, _):
			variables = state
			next_var = evaluate_fn(rpn_expr, rpn_consts, variables)
			new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
			return new_state, next_var

		_, output = jax.lax.scan(step_fn, inputs, length=O)

		rpn_pad_mask = rpn_token_string != self.token_map['PAD']
		inputs_pad_mask = jnp.arange(I) > I - rpn_degree - 1
		output_pad_mask = jnp.full((O,), True)

		series = jnp.concatenate((inputs, output))
		# jax.debug.print("inputs: {}\nrpn_consts: {}\ntokens: {}\nrpn_expr: {}\nseries: {}", 
				  # inputs, rpn_consts, tokens, rpn_expr, series)

		series_pad_mask = jnp.concatenate((inputs_pad_mask, output_pad_mask))

		series_enc, series_positions = jfuncs.tokenize_ints(
			series,
			series_pad_mask,
			self.opts.int_base,
			self.token_map['0'],
			self.token_map['PLUS_SIGN'],
			self.token_map['MINUS_SIGN'],
			self.token_map['PAD']
		)
		series_enc_pad_mask = series_positions > -1
		obs_sym_pad_mask = jnp.concatenate((rpn_pad_mask, series_enc_pad_mask))
		target_pad_mask = jnp.concatenate((rpn_pad_mask, series_positions < rpn_degree))
		obs_sym = jnp.concatenate((rpn_token_string, series_enc))
		obs_sym, obs_sym_ntok = jfuncs.compact_masked(obs_sym, obs_sym_pad_mask)
		input_mask = jnp.arange(obs_sym.shape[0]) < obs_sym_ntok
		target_mask = jfuncs.compact_masked(target_pad_mask, obs_sym_pad_mask)

		return obs_sym, input_mask, target_mask 

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		B = key_B.shape[0]
		obs_sym_BC, input_mask_BC, target_mask_BC = jax.vmap(self._generate_one)(key_B)
		return TokensAndProbs(
				jax.random.key_data(key_B), 
				obs_sym=obs_sym_BC,
				obs_prob=None,
				input_mask=input_mask_BC,
				target_mask=target_mask_BC) 


