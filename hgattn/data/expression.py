import random
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Union, Iterable

class BinaryOp(Enum):
	ADD = "+"
	SUB = "-" 
	MUL = "*" 
	INTDIV = "//" 
	MOD = "mod" 

class UnaryOp(Enum):
	ABS = "abs" 
	SQR = "sqr" 
	SIGN = "sign" 
	RELU = "relu" 

@dataclass(frozen=True)
class Literal:
	value: int | str

@dataclass(frozen=True)
class UnaryExpr:
	op: UnaryOp
	operand: 'Node'

@dataclass(frozen=True)
class BinaryExpr:
	op: BinaryOp
	left: 'Node'
	right: 'Node'

Node = Union[Literal, UnaryExpr, BinaryExpr]

def get_variables(n: int):
	return tuple(chr(ord('A') + i) for i in range(n))

def gen_expressions(
	depth: int, 
	binops: list[BinaryOp], 
	uops: list[UnaryOp],
	const_beg: int,
	const_end: int,
	n_const_samples: int,
	variables: list[str],
) -> Iterable[Node]:
	"""
	Generate all expressions of a given depth and up to n_vars.
	Do not enumerate all possible constants at each 
	"""
	consts = range(const_beg, const_end)

	def _gen(depth) -> Iterable[Node]:
		if depth == 0:
			yield from [Literal(v) for v in variables]
			yield from [Literal(c) for c in np.random.choice(consts, n_const_samples).tolist()]
			return

		yield from [
				BinaryExpr(op, l, r) 
				for op in binops 
				for l in _gen(depth-1)
				for r in _gen(depth-1) 
				]

		yield from [
				UnaryExpr(op, val)
				for op in uops
				for val in _gen(depth-1)
				]

	return _gen(depth)

RPN = list[str|UnaryOp|BinaryOp]

def node_to_rpn(node: Node) -> RPN:
	"""
	Convert the Node to Reverse Polish Notation representation
	"""
	rpn = []
	def _visit(prefix: list[str], node: Node):
		match node:
			case Literal(val):
				prefix.append(val)
			case UnaryExpr(op, operand):
				_visit(prefix, operand)
				prefix.append(op)
			case BinaryExpr(op, left, right):
				_visit(prefix, left)
				_visit(prefix, right)
				prefix.append(op)
			case default:
				raise RuntimeError(f"Unexpected node type: {type(node)}")
	_visit(rpn, node)
	return rpn

def evaluate(rpn: RPN, **binds) -> int:
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
	n_const_samples: int # number of individual samples to take from [const_beg, const_end)
	n_vars: int          # number of different variables that can appear
	
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

		if self.n_const_samples > len(range(self.const_beg, self.const_end)):
			raise RuntimeError(
				f"Received {self.n_const_samples=}, which exceeds the range of allowable "
				f"constants [{self.const_beg}, {self.const_end})")
		if self.n_vars > 10:
			raise RuntimeError(f"Received {self.n_vars=} exceeding max of 10")

class InductiveDataset(eqx.Module):
	opts: InductiveOpts = eqx.field(static=True)
	rpn_exprs: jax.Array

	def __init__(self, opts: InductiveOpts):
		self.opts = opts
		variables = get_variables(opts.n_vars)

		exprs = list(gen_expressions(
			opts.max_expr_depth,
			opts.binops,
			opts.uops,
			opts.const_beg,
			opts.const_end,
			opts.n_const_samples,
			variables))

		if len(exprs) < opts.n_exprs:
			raise RuntimeError(
					f"Generated only {len(exprs)} expressions but require {opts.n_exprs}")
		token_vals = ('PAD', *opts.binops, *opts.uops, *variables, *range(opts.int_base))
		token_map = { val: idx for idx, val in enumerate(token_vals) }
		random.shuffle(exprs)
		exprs = exprs[:opts.n_exprs]
		rpns = tuple(node_to_rpn(expr) for expr in exprs)
		max_rpn_len = max(len(rpn) for rpn in rpns)
		rpn_ary = np.full((opts.n_exprs, max_rpn_len), token_map['PAD'], dtype=np.int32)
		for idx, rpn in enumerate(rpns):
			rpn_ary[idx,:len(rpn)] = tuple(token_map[t] for t in rpn)

		self.rpn_exprs = jnp.array(rpn_ary)



