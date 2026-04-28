import numpy as np
from typing import Union, Iterable, Self, Callable
from enum import Enum
from dataclasses import dataclass
import operator

class BinaryOp(Enum):
	ADD = "add"
	SUB = "sub" 
	MUL = "mul" 
	INTDIV = "intdiv" 
	MOD = "mod" 
	MOD_ADD = "mod_add"
	MOD_SUB = "mod_sub"
	MOD_MUL = "mod_mul"
	MOD_INTDIV = "mod_intdiv"

class UnaryOp(Enum):
	ABS = "abs" 
	SQR = "sqr" 
	SIGN = "sign" 
	RELU = "relu" 
	MOD_SQR = "mod_sqr"

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
RPNValue = Union[BinaryOp, UnaryOp, str, int]

class RPNExpression:
	def __init__(self, node: Node, mod_val: int):
		def _postorder(node):
			match node:
				case Variable() | Const():
					yield node
				case UnaryExpr(op, operand):
					yield from _postorder(operand)
					yield op
				case BinaryExpr(op, left, right):
					yield from _postorder(left)
					yield from _postorder(right)
					yield op
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		self.root = node
		self.mod_val = mod_val
		self.nodes = tuple(_postorder(node))

	def get_binop(self, op: BinaryOp) -> Callable[[int, int], int]:
		return {
				BinaryOp.ADD: operator.add,
				BinaryOp.SUB: operator.sub,
				BinaryOp.MUL: operator.mul,
				BinaryOp.INTDIV: lambda x, y: 0 if y == 0 else operator.floordiv,
				BinaryOp.MOD: operator.mod,
				BinaryOp.MOD_ADD: lambda x, y: (x + y) % self.mod_val,
				BinaryOp.MOD_SUB: lambda x, y: (x - y) % self.mod_val,
				BinaryOp.MOD_MUL: lambda x, y: (x * y) % self.mod_val,
				BinaryOp.MOD_INTDIV: lambda x, y: 0 if y == 0 else (x // y) % self.mod_val 
				}[op]
	
	def get_uop(self, op: UnaryOp) -> Callable[[int], int]:
		return {
				UnaryOp.ABS: abs,
				UnaryOp.SQR: lambda x: pow(x, 2),
				UnaryOp.SIGN: lambda x: -1 if x < 0 else 1,
				UnaryOp.RELU: lambda x: max(0, x),
				UnaryOp.MOD_SQR: lambda x: pow(x, 2) % self.mod_val
				}[op]

	@classmethod
	def from_vals(cls, vals: list[RPNValue], mod_val: int) -> Self:
		assert len(vals) > 0, "cannot convert empty expression"
		stack = []
		for val in vals:
			match val:
				case int(i):
					stack.append(Const(i))
				case str(s):
					stack.append(Variable(s))
				case BinaryOp():
					try:
						r = stack.pop()
						l = stack.pop()
						stack.append(BinaryExpr(val, l, r))
					except IndexError:
						raise RuntimeError(f"stack empty:  invalid RPN expression")
				case UnaryOp():
					try:
						v = stack.pop()
						stack.append(UnaryExpr(val, v))
					except IndexError:
						raise RuntimeError(f"stack empty:  invalid RPN expression")
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		if len(stack) != 1:
			raise RuntimeError(f"Invalid RPN expression: len(stack) != 1 at end")
		return cls(stack.pop(), mod_val)

	def __repr__(self):
		strs = []
		for v in self.token_vals:
			match v:
				case BinaryOp() | UnaryOp():
					strs.append(v.value)
				case str() | int():
					strs.append(str(v))
		return ' '.join(strs)

	@property
	def variables(self):
		return tuple(sorted(set(v.value for v in self.nodes if isinstance(v, Variable))))

	@property
	def const_values(self):
		return tuple(sorted(set(c.value for c in self.nodes if isinstance(c, Const))))

	@property
	def token_vals(self) -> list[RPNValue]:
		_toks = []
		for node in self.nodes:
			match node:
				case Variable(val) | Const(val):
					_toks.append(val)
				case UnaryOp():
					_toks.append(node)
				case BinaryOp():
					_toks.append(node)
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		return tuple(_toks)

	def evaluate(self, **binds) -> int:
		stack = []
		for node in self.nodes:
			match node:
				case Variable(name):
					val = binds.get(name)
					if val is None:
						raise RuntimeError(f"Variable {name} found but missing bind")
					stack.append(val)
				case Const(val):
					stack.append(val)
				case UnaryOp():
					try:
						val = stack.pop()
					except IndexError:
						raise RuntimeError(f"Got Unary op {op.value} but stack is empty")
					op = self.get_uop(node)
					stack.append(op(val))
				case BinaryOp():
					try:
						rval = stack.pop()
						lval = stack.pop()
					except IndexError:
						raise RuntimeError(f"Got Unary op {op.value} but stack is empty")
					op = self.get_binop(node)
					stack.append(op(lval, rval))
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		if len(stack) != 1:
			raise RuntimeError(f"stack length is {len(stack)} at end of input. Should be 1")
		return stack.pop()
	

def gen_expressions(
	seed: int,
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
	rng = np.random.default_rng(seed=seed)
	ops = np.array(binops + uops)
	O = ops.shape[0]

	def _gen(depth, nc, nv) -> Iterable[tuple[Node, int, int]]:
		if depth == 0:
			if nv > 0:
				ary = np.array(variables)
				v = rng.choice(ary, 1).item()
				yield Variable(v), nc, nv - 1
			if nc > 0:
				c = rng.choice(consts, 1).item()
				yield Const(c), nc - 1, nv
			return

		inds = np.mgrid[:O, :depth, :depth].T.reshape(-1, 3)
		for op_ind, ldepth, rdepth in rng.permutation(inds): 
			op = ops[op_ind]
			match op:
				case BinaryOp():
					for l, nc1, nv1 in _gen(ldepth, nc, nv):
						for r, nc2, nv2 in _gen(rdepth, nc1, nv1):
							yield BinaryExpr(op, l, r), nc2, nv2
				case UnaryOp():
					for val, nc1, nv1 in _gen(ldepth, nc, nv):
						yield UnaryExpr(op, val), nc1, nv1
				case _:
					raise RuntimeError("Unknown op: {op}")

	for node, _, _ in _gen(depth, n_vars, n_consts):
		yield node

