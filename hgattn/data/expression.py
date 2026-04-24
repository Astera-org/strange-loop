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
from .types import TokensAndProbs

class BinaryOp(Enum):
	ADD = "add"
	SUB = "sub" 
	MUL = "mul" 
	INTDIV = "intdiv" 
	MOD = "mod" 

class UnaryOp(Enum):
	ABS = "abs" 
	SQR = "sqr" 
	SIGN = "sign" 
	RELU = "relu" 

@dataclass(frozen=True)
class Variable:
	value: str

@dataclass(frozen=True)
class Const:
	value: int 

@dataclass(frozen=True)
class UnaryExpr:
	op: UnaryOp
	operand: 'Node'

@dataclass(frozen=True)
class BinaryExpr:
	op: BinaryOp
	left: 'Node'
	right: 'Node'

Node = Union[Variable, Const, UnaryExpr, BinaryExpr]

def get_variables(n: int):
	return tuple(chr(ord('A') + i) for i in range(n))

def get_constants(n: int):
	return tuple('c' + chr(ord('A') + i) for i in range(n))

def gen_expressions(
	depth: int, 
	binops: list[BinaryOp], 
	uops: list[UnaryOp],
	n_consts: int,
	n_vars: int,
	consts: list[int],
	variables: list[str],
) -> Iterable[Node]:
	"""
	Generate all expressions of a given depth and up to n_vars.
	Do not enumerate all possible constants at each 
	"""
	def _gen(depth, nc, nv) -> Iterable[tuple[Node, int, int]]:
		if depth == 0:
			if nv > 0:
				for v in variables:
					yield Variable(v), nc, nv - 1
			if nc > 0:
				for c in np.random.choice(consts, nc).tolist():
					yield Const(c), nc - 1, nv
			return

		for op in binops:
			for l, nc1, nv1 in _gen(depth-1, nc, nv):
				for r, nc2, nv2 in _gen(depth-1, nc1, nv1):
					yield BinaryExpr(op, l, r), nc2, nv2

		for op in uops:
			for val, nc1, nv1 in _gen(depth-1, nc, nv):
				yield UnaryExpr(op, val), nc1, nv1

	for node, _, _ in _gen(depth, n_vars, n_consts):
		yield node

RPN = list[str|UnaryOp|BinaryOp]

@dataclass
class RPNExpr:
	tokens: list[str]
	consts: list[int]

def node_to_rpn(node: Node) -> RPNExpr:
	"""
	Convert the Node to Reverse Polish Notation representation
	"""
	tokens = []
	consts = {} # 'cA' => int
	it = iter(get_constants(10))

	def _visit(prefix: list[str], node: Node):
		match node:
			case Variable(val):
				prefix.append(val)
			case Const(val):
				consts.setdefault(val, next(it))
				prefix.append(consts[val])
			case UnaryExpr(op, operand):
				_visit(prefix, operand)
				prefix.append(op.value)
			case BinaryExpr(op, left, right):
				_visit(prefix, left)
				_visit(prefix, right)
				prefix.append(op.value)
			case _:
				raise RuntimeError(f"Unexpected node type: {type(node)}")
	_visit(tokens, node)
	return RPNExpr(tokens, list(sorted(consts.keys())))

def evaluate_py(rpn: RPN, **binds) -> int:
	"""
	Idea is to maintain a stack while traversing the RPN expression.
	Values in the RPN are tokens, while values in the Stack will be int64 values.
	The final output will be 

	RPN                 Stack
	3 8 * 3 5 * +       -
	  8 * 3 5 * +       3
	    * 3 5 * +       3 8
	      3 5 * +       24
	        5 * +       24 3
	          * +       24 3 5
	            +       24 15
	                    39
	"""
	stack = []
	binops = {
			BinaryOp.ADD: operator.add,
			BinaryOp.SUB: operator.sub,
			BinaryOp.MUL: operator.mul,
			BinaryOp.INTDIV: operator.floordiv,
			BinaryOp.MOD: operator.mod
			}
	uops = {
			UnaryOp.ABS: abs,
			UnaryOp.SQR: lambda x: pow(x, 2),
			UnaryOp.SIGN: lambda x: -1 if x < 0 else 1,
			UnaryOp.RELU: lambda x: max(0, x),
			}

	for tok in rpn:
		match tok:
			case int():
				stack.append(tok)
			case str():
				bind = binds.get(tok)
				if bind is None:
					raise RuntimeError(f"no value supplied for variable {tok}")
				stack.append(bind)
			case UnaryOp():
				op = uops.get(tok)
				if op is None:
					raise RuntimeError(f"unrecognized unary op: {tok}")
				try:
					val = stack.pop()
				except IndexError:
					raise RuntimeError(f"invalid RPN: not enough values for unary op")
				stack.append(op(val))
			case BinaryOp():
				op = binops.get(tok)
				if op is None:
					raise RuntimeError(f"Unrecognized binary op: {tok}")
				try:
					right = stack.pop()
					left = stack.pop()
				except IndexError:
					raise RuntimeError(f"invalid RPN: not enough values for binary op")
				val = op(left, right)
				stack.append(val)
			case _:
				raise RuntimeError(f"Unrecognized token type: {type(tok)}")

	if len(stack) != 1:
		raise RuntimeError(f"Invalid RPN: stack length is {len(stack)} but should be 1")
	return stack.pop()



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
	rpn_consts: jax.Array

	def __init__(self, opts: InductiveOpts):
		self.opts = opts
		all_ops = tuple([op.value for op in BinaryOp] + [op.value for op in UnaryOp])
		all_const_vals = list(range(opts.const_beg, opts.const_end))
		variables = get_variables(opts.n_vars)
		constants = get_constants(opts.n_consts)
		# import pdb
		# pdb.set_trace()

		digits = tuple(str(d) for d in range(opts.int_base))
		rpn_tokens = ('PAD',) + all_ops + variables + constants 
		sym_tokens = ('PAD',) + all_ops + variables + ('PLUS_SIGN', 'MINUS_SIGN') + digits
		self.rpn_token_map = { tok: idx for idx, tok in enumerate(rpn_tokens) }
		self.sym_token_map = { tok: idx for idx, tok in enumerate(sym_tokens) }
		self.vocab_size = len(self.sym_token_map)

		exprs = list(
				gen_expressions(
					opts.max_expr_depth,
					opts.binops,
					opts.uops,
					opts.n_consts,
					opts.n_vars,
					all_const_vals,
					variables)
				)

		if len(exprs) < opts.n_exprs:
			raise RuntimeError(
					f"Generated only {len(exprs)} expressions but require {opts.n_exprs}")

		exprs = random.sample(exprs, opts.n_exprs)
		rpns = tuple(node_to_rpn(expr) for expr in exprs)
		max_rpn_len = max(len(rpn.tokens) for rpn in rpns)
		PAD = self.rpn_token_map['PAD']
		rpn_ary = np.full((opts.n_exprs, max_rpn_len), PAD, dtype=np.int32)
		rpn_consts = np.full((opts.n_exprs, len(constants)), PAD, dtype=np.int32)

		for idx, rpn in enumerate(rpns):
			toks = rpn.tokens
			rpn_ary[idx,:len(rpn.tokens)] = tuple(self.rpn_token_map[t] for t in toks)
			rpn_consts[idx,:len(rpn.consts)] = rpn.consts

		self.rpn_exprs = jnp.array(rpn_ary)
		self.rpn_consts = jnp.array(rpn_consts)

	@eqx.filter_jit
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
				binary(jnp.add),
				binary(jnp.subtract),
				binary(jnp.multiply),
				binary(jnp.floor_divide),
				binary(jnp.mod),
				unary(jnp.abs),
				unary(lambda x: jnp.power(x, 2)),
				unary(lambda x: jnp.where(x >= 0, 1, -1)),
				unary(lambda x: jnp.maximum(0, x))
			]

			for i in range(self.opts.n_vars):
				branches.append(push(lambda i=i: push(variables[i])))
			
			for i in range(self.opts.n_consts):
				branches.append(push(lambda i=i: push(constants[i])))

			new_stack, new_ptr = jax.lax.switch(rpn_token, branches)
			return (new_stack, new_ptr, constants, variables), None

		def evaluate_rpn(rpn_expr, rpn_consts, variables):
			stack = jnp.array((self.opts.max_expr_depth,), dtype=jnp.int64)
			ptr = jnp.array(0, dtype=jnp.int32)
			state = stack, ptr, rpn_consts, variables
			final_state, _ = jax.lax.scan(rpn_step, state, rpn_expr)
			final_stack = final_state[0]
			return final_stack[0]

		def generate_one(key):
			expr_key, input_key = jax.random.split(key)
			expr_index = jax.random.choice(expr_key, self.opts.n_exprs)
			rpn_expr = self.rpn_exprs[expr_index,:]
			rpn_consts = self.rpn_consts[expr_index,:]
			inputs = jax.random.choice(
					input_key, 
					jnp.arange(self.opts.input_beg, self.opts.input_end),
					(self.opts.n_vars,))

			def step_fn(state, _):
				variables = state
				next_var = evaluate_rpn(rpn_expr, rpn_consts, variables)
				new_state = jnp.roll(variables, -1, 0).at[-1].set(next_var)
				return new_state, next_var

			_, output = jax.lax.scan(step_fn, inputs, length=self.opts.n_outputs)
			tokens = jfuncs.tokenize_input(
					output, self.opts.int_base, self.zero_token, self.plus_token, 
					self.minus_token)
			return tokens

		outputs_BC = jax.vmap(generate_one)(key_B)
		return TokensAndProbs(jax.random.key_data(key_B), obs_sym=outputs_BC) 

