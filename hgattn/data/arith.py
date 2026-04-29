import numpy as np
from typing import Union, Iterable, Self, Callable
from enum import Enum
from dataclasses import dataclass
import random
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

	def infix(self) -> str:
		opstrings = {
				BinaryOp.ADD: "+",
				BinaryOp.SUB: "-",
				BinaryOp.MUL: "*",
				BinaryOp.INTDIV: "//",
				BinaryOp.MOD: "%",
				BinaryOp.MOD_ADD: "+",
				BinaryOp.MOD_SUB: "-",
				BinaryOp.MOD_MUL: "*",
				BinaryOp.MOD_INTDIV: "//",
				UnaryOp.ABS: "abs",
				UnaryOp.SQR: "sqr",
				UnaryOp.SIGN: "sign",
				UnaryOp.RELU: "relu",
				UnaryOp.MOD_SQR: "mod_sqr",
		}

		def _rec(node):
			match node:
				case Variable(name):
					return name
				case Const(val):
					return val
				case UnaryExpr(op, sub):
					sub_str = _rec(sub)
					return f"{opstrings[op]}({sub_str})"
				case BinaryExpr(op, left, right):
					left_str = _rec(left)
					right_str = _rec(right)
					return f"({left_str} {opstrings[op]} {right_str})"
		return _rec(self.root)
	

class TreeGen:
	def __init__(
		self, 
		binops: list[BinaryOp], 
		uops: list[UnaryOp], 
		variables: list[str], 
		consts: list[int]
	):
		self.binops = tuple(binops)
		self.uops = tuple(uops)
		self.variables = tuple(variables)
		self.consts = tuple(consts)

	@property
	def leaf_vals(self):
		return self.variables + self.consts

	def _get_labeled_counts(self, max_depth):
		n_uops = len(self.uops)
		n_binops = len(self.binops)
		n_leaf = len(self.variables) + len(self.consts)

		counts = [n_leaf]
		for i in range(1, max_depth + 1):
			t0 = n_leaf
			t1 = n_uops * counts[i-1]
			t2 = n_binops * (counts[i-1] ** 2)
			counts.append(t0 + t1 + t2)
		return counts

	def _unrank_labeled(self, index, depth, counts):
		n_uops = len(self.uops)
		n_binops = len(self.binops)
		n_leaf = len(self.leaf_vals) 
		if depth == 0 or index < n_leaf:
			val = self.leaf_vals[index]
			match val:
				case str(s):
					return Variable(s)
				case int(i):
					return Const(i)
				case _:
					raise RuntimeError(f"Unknown value type: {val}.  Should be int or str")
		index -= n_leaf
		prev_count = counts[depth-1]
		unary_total = n_uops * prev_count
		if index < unary_total:
			op_idx, child_idx = divmod(index, prev_count)
			child = self._unrank_labeled(child_idx, depth - 1, counts)
			return UnaryExpr(self.uops[op_idx], child)

		index -= unary_total
		op_idx, combined_idx = divmod(index, prev_count**2)
		left_idx, right_idx = divmod(combined_idx, prev_count)

		left = self._unrank_labeled(left_idx, depth - 1, counts)
		right = self._unrank_labeled(right_idx, depth - 1, counts)
		return BinaryExpr(self.binops[op_idx], left, right)

	def gen_trees(self, seed: int, max_depth: int, n: int) -> list[Node]:
		"""
		Generate n trees of up to `max_depth` depth sampled uniformly
		from all possible trees using the binops, uops, variables and consts.
		"""
		counts = self._get_labeled_counts(max_depth)
		total_trees = counts[max_depth]
		rng = random.Random(seed)

		if n > total_trees:
			raise RuntimeError(f"{n=} exceeds total trees {total_trees}")

		def _get_indices(total_trees, n):
			indices = set()
			while len(indices) < n:
				indices.add(rng.randrange(total_trees))
			return list(indices)

		indices = _get_indices(total_trees, n)
		return [self._unrank_labeled(i, max_depth, counts) for i in indices]

