import operator
import numpy as np
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

def gen_expressions(
	depth: int, 
	binops: list[BinaryOp], 
	uops: list[UnaryOp],
	const_start: int,
	const_end: int,
	num_const_samples: int,
	num_vars: int
) -> Iterable[Node]:
	"""
	Generate all expressions of a given depth and up to num_vars.
	Do not enumerate all possible constants at each 
	"""
	assert num_vars <= 10, f"num_vars must be <= 10, got {num_vars}"

	variables = "ABCDEFGHIJ"[:num_vars]
	consts = range(const_start, const_end)

	def _gen(depth) -> Iterable[Node]:
		if depth == 0:
			yield from [Literal(v) for v in variables]
			yield from [Literal(c) for c in np.random.choice(consts, num_const_samples).tolist()]
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
				val = op(stack.pop())
				stack.append(val)
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

