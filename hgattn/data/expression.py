import random
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Union, Iterable
from jaxtyping import PRNGKeyArray
from .. import jfuncs
from . import arith
from .arith import BinaryOp, UnaryOp
from .types import TokensAndProbs

def get_variables(n: int):
	return tuple(reversed(tuple(chr(ord('A') + i) for i in range(n))))

def get_constants(n: int):
	return tuple('c' + chr(ord('A') + i) for i in range(n))


@dataclass
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

		if self.n_consts > len(range(self.const_beg, self.const_end)):
			raise RuntimeError(
				f"Received {self.n_consts=}, which exceeds the range of allowable "
				f"constants [{self.const_beg}, {self.const_end})")
		if self.n_vars > 10:
			raise RuntimeError(f"Received {self.n_vars=} exceeding max of 10")

class InductiveDataset(eqx.Module):
	opts: InductiveOpts = eqx.field(static=True)
	vocab_size: int = eqx.field(static=True)
	rpn_token_map: dict[str, int] = eqx.field(static=True)
	sym_token_map: dict[str, int] = eqx.field(static=True)
	rpn_exprs: jax.Array
	rpn_tokens: jax.Array
	rpn_consts: jax.Array
	rpn_degree: jax.Array

	def __init__(self, opts: InductiveOpts):
		jax.config.update("jax_enable_x64", True)
		self.opts = opts
		all_ops = tuple([op.value for op in BinaryOp] + [op.value for op in UnaryOp])
		all_const_vals = list(range(opts.const_beg, opts.const_end))
		variables = get_variables(opts.n_vars)
		constants = get_constants(opts.n_consts)

		digits = tuple(str(d) for d in range(opts.int_base))
		controls = 'PLUS_SIGN', 'MINUS_SIGN', 'EQUALS'
		rpn_tokens = ('PAD',) + all_ops + variables + constants 
		sym_tokens = ('PAD',) + all_ops + variables + controls + digits
		self.rpn_token_map = { tok: idx for idx, tok in enumerate(rpn_tokens) }
		self.sym_token_map = { tok: idx for idx, tok in enumerate(sym_tokens) }
		self.vocab_size = len(self.sym_token_map)

		nodes = list(
				arith.gen_expressions(
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

		def _to_tokens(rpn):
			toks = []
			for v in rpn.token_vals:
				match v:
					case int(i):
						ci = rpn.consts.index(i)
						toks.append(self.rpn_token_map[constants[ci]])
					case str(s):
						toks.append(self.rpn_token_map[s])
			return np.array(toks)

		def _to_symbols(rpn):
			syms = []
			for v in rpn.token_vals:
				match v:
					case int(i): 
						enc = jfuncs.tokenize_one_int(
								i, opts.int_base, 
								self.sym_token_map['0'],
								self.sym_token_map['PLUS_SIGN'],
								self.sym_token_map['MINUS_SIGN'])
						syms.extend(enc.tolist())
					case str(s):
						syms.append(self.sym_token_map[s])
			return np.array(syms) 

		def ragged_stack(arrays, pad):
			N = len(arrays)
			D = max(len(a) for a in arrays)
			result = np.full((N, D), pad, dtype=arrays[0].dtype)
			for idx, ary in enumerate(arrays):
				result[idx,:len(ary)] = ary
			return jnp.array(result)

		rpns = tuple(arith.RPNExpression(n) for n in nodes)
		rpns = tuple(rpn for rpn in rpns if len(rpn.variables) > 0)
		rpns = random.sample(rpns, opts.n_exprs)

		PAD = self.rpn_token_map['PAD']
		self.rpn_exprs = ragged_stack([_to_tokens(rpn) for rpn in rpns], PAD)
		self.rpn_tokens = ragged_stack([_to_symbols(rpn) for rpn in rpns], PAD)
		self.rpn_consts = ragged_stack([np.array(rpn.consts) for rpn in rpns], PAD)

		_deg = lambda rpn: len(variables) - variables.index(rpn.variables[-1])
		self.rpn_degree = jnp.array([_deg(rpn) for rpn in rpns])

	# @eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		B = key_B.shape[0]
	
		def rpn_step(state, rpn_token):
			stack, ptr, constants, variables = state

			def push(val):
				return stack.at[ptr].set(val), ptr + 1

			def binary(op_func):
				a, b = stack[ptr-2], stack[ptr-1]
				return stack.at[ptr-2].set(op_func(a, b)), ptr - 1

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

			for i in range(self.opts.n_vars):
				branches.append(lambda i=i: push(variables[i]))
			
			for i in range(self.opts.n_consts):
				branches.append(lambda i=i: push(constants[i]))

			new_stack, new_ptr = jax.lax.switch(rpn_token, branches)
			return (new_stack, new_ptr, constants, variables), None

		def evaluate_rpn(rpn_expr, rpn_consts, variables):
			stack = jnp.empty((self.opts.max_expr_depth,), dtype=jnp.int64)
			ptr = jnp.array(0, dtype=jnp.int32)
			state = stack, ptr, rpn_consts, variables
			final_state, _ = jax.lax.scan(rpn_step, state, rpn_expr)
			final_stack = final_state[0]
			return final_stack[0]

		def generate_one(key):
			expr_key, input_key = jax.random.split(key)
			expr_index = jax.random.choice(expr_key, self.opts.n_exprs)
			rpn_expr = self.rpn_exprs[expr_index,:]
			rpn_token_string = self.rpn_tokens[expr_index,:]
			rpn_consts = self.rpn_consts[expr_index,:]
			rpn_degree = self.rpn_degree[expr_index]
			V, O = self.opts.n_vars, self.opts.n_outputs
			rpn_mask = rpn_token_string != self.sym_token_map['PAD']
			inputs_mask = jnp.arange(V, 0, -1) <= rpn_degree # TODO: check this
			output_mask = jnp.full((O,), True)

			inputs = jax.random.choice(
					input_key, 
					jnp.arange(self.opts.input_beg, self.opts.input_end),
					(V,))

			pre_series_mask = jnp.concatenate((inputs_mask, output_mask))

			def step_fn(state, _):
				variables = state
				next_var = evaluate_rpn(rpn_expr, rpn_consts, variables)
				new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
				return new_state, next_var

			_, output = jax.lax.scan(step_fn, inputs, length=self.opts.n_outputs)
			series = jnp.concatenate((inputs, output))
			series, _ = jfuncs.compact_masked(series, pre_series_mask)
			series_mask = jnp.arange(V + O) >= rpn_degree 

			full_mask = jnp.concatenate((rpn_mask, series_mask))
			full = jnp.concatenate((rpn_token_string, series))
			obs_sym, _ = jfuncs.compact_masked(full, full_mask)

			import pdb
			pdb.set_trace()

			return obs_sym, input_mask 

			"""
			tokens, mask = jfuncs.tokenize_ints(
					series,
					series_mask
					output, 
					self.opts.int_base,
					self.sym_token_map['0'],
					self.sym_token_map['PLUS_SIGN'],
					self.sym_token_map['MINUS_SIGN'])
			return tokens, mask, rpn_expr, rpn_consts
			"""

		series_BC, series_mask_BC, rpn_expr_BC, rpn_consts_BC = jax.vmap(generate_one)(key_B)
		return TokensAndProbs(
				jax.random.key_data(key_B), 
				obs_sym=series_BC,
				obs_prob=None,
				input_mask=series_mask_BC,
				target_mask=rpn_consts_BC) 

