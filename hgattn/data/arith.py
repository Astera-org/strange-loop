import numpy as np
from typing import Union, Iterable
from enum import Enum
from dataclasses import dataclass

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

class RPNExpression:
	def __init__(self, node: Node):
		def _postorder(node):
			match node:
				case Variable() | Const():
					yield node
				case UnaryExpr(op, operand):
					yield from _postorder(operand)
					yield operand
				case BinaryExpr(op, left, right):
					yield from _postorder(left)
					yield from _postorder(right)
					yield op
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		self.nodes = tuple(_postorder(node))

	@property
	def variables(self):
		return tuple(sorted(set(v.value for v in self.nodes if isinstance(v, Variable))))

	@property
	def consts(self):
		return tuple(sorted(set(c.value for c in self.nodes if isinstance(c, Const))))

	@property
	def token_vals(self):
		_toks = []
		for node in self.nodes:
			match node:
				case Variable(val) | Const(val):
					_toks.append(val)
				case UnaryOp():
					_toks.append(node.value)
				case BinaryOp():
					_toks.append(node.value)
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		return tuple(_toks)
	

def evaluate_py(rpn: RPNExpression, **binds) -> int:
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

