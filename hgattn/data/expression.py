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

def node_to_rpn(node: Node) -> list[str]:
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
				prefix.append(op.value)
			case BinaryExpr(op, left, right):
				_visit(prefix, left)
				_visit(prefix, right)
				prefix.append(op.value)
			case default:
				raise RuntimeError(f"Unexpected node type: {type(node)}")
	_visit(rpn, node)
	return rpn




