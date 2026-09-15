import numpy as np
import math
from typing import Union, Iterable, Self, Callable
import jax
import jax.numpy as jnp
from enum import Enum
from dataclasses import dataclass
import random
import operator
from collections import Counter
from .mathops import BinaryOp, UnaryOp


@dataclass(frozen=True)
class Variable:
	name: str

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
RPNValue = Union[BinaryOp, UnaryOp, Variable, Const] 

def parse_rpn_value(val: int|str) -> RPNValue:
	match val:
		case int():
			return Const(val)
		case str():
			for cls in (BinaryOp, UnaryOp, Variable):
				try:
					return cls(val)
				except ValueError:
					pass
		case _:
			raise RuntimeError(f"Can't parse {val} of type {type(val)} into RPNValue")

class RPNExpression:
	def __init__(self, node: Node, mod_val: int):
		def _postorder(node):
			match node:
				case Variable() | Const():
					yield node
				case UnaryExpr(op, operand):
					yield from _postorder(operand)
					yield node
				case BinaryExpr(op, left, right):
					yield from _postorder(left)
					yield from _postorder(right)
					yield node 
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		def _depth(node):
			match node:
				case Variable() | Const():
					return 0
				case UnaryExpr(_, operand):
					return _depth(operand) + 1
				case BinaryExpr(_, left, right):
					return max(_depth(left), _depth(right)) + 1
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")

		self.root = node
		self.mod_val = 2**64 if mod_val is None else mod_val
		self.nodes = tuple(_postorder(node))
		self.depth = _depth(node) 
		self.variable_names = tuple(set([n.name for n in self.nodes if isinstance(n, Variable)]))

	def get_binop(self, op: BinaryOp) -> Callable[[int, int], int]:
		def _intdiv(x, y):
			mask = (y != 0)
			dest = np.zeros_like(y)
			dest[mask] = x[mask] // y[mask]
			return dest

		return {
				BinaryOp.ADD: lambda x, y: (x + y) % self.mod_val,
				BinaryOp.MUL: lambda x, y: (x * y) % self.mod_val,
				BinaryOp.INTDIV: lambda x, y: _intdiv(x, y),
				BinaryOp.MOD: lambda x, y: x % y,
				}[op]
	
	def get_uop(self, op: UnaryOp) -> Callable[[int], int]:
		return {
				UnaryOp.ABS: abs,
                UnaryOp.POW2: lambda x: (x ** 2) % self.mod_val,
                UnaryOp.POW3: lambda x: (x ** 3) % self.mod_val,
				UnaryOp.SIGN: lambda x: np.where(x >= 0, 1, -1),
				UnaryOp.RELU: lambda x: np.maximum(0, x),
				}[op]

	@classmethod
	def from_vals(cls, vals: list[RPNValue], mod_val: int) -> Self:
		assert len(vals) > 0, "cannot convert empty expression"
		stack = []
		for val in vals:
			match val:
				case Variable() | Const():
					stack.append(val)
				case BinaryOp():
					try:
						r = stack.pop()
						l = stack.pop()
						stack.append(BinaryExpr(val, l, r))
					except IndexError:
						raise RuntimeError(f"stack empty:  invalid RPN expression: {vals}")
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

	@property
	def token_vals(self) -> list[RPNValue]:
		_toks = []
		for node in self.nodes:
			match node:
				case Variable(val) | Const(val):
					_toks.append(val)
				case UnaryExpr(op):
					_toks.append(op)
				case BinaryExpr(op):
					_toks.append(op)
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		return tuple(_toks)

	def __repr__(self):
		strs = []
		for v in self.token_vals:
			match v:
				case BinaryOp() | UnaryOp():
					strs.append(v.name)
				case str(v):
					strs.append(v)
				case int(i):
					strs.append(str(i))
				case _:
					raise RuntimeError(f"Unexpected token val: {v}")
		return ' '.join(strs)

	def evaluate(self, **binds) -> np.array:
		binds = { k: np.array(v) for k, v in binds.items() }
		arg0 = next(iter(binds.values()))
		stack_depth = self.depth + 2
		stack = np.empty((stack_depth, *arg0.shape), dtype=arg0.dtype)
		ptr = np.array(0)

		def push(st, ptr, val):
			st[ptr] = val
			ptr += 1

		def binary(st, ptr, func):
			if ptr < 2:
				raise RuntimeError(f"Got binary op but stack has less than two elements")
			l, r = st[ptr-2], st[ptr-1]
			st[ptr-2] = func(l, r)
			ptr -= 1

		def unary(st, ptr, func):
			if ptr < 1:
				raise RuntimeError(f"Got unary op but stack is empty")
			a = st[ptr-1]
			st[ptr-1] = func(a)

		for node in self.nodes:
			match node:
				case Variable(name):
					val = binds.get(name)
					if val is None:
						raise RuntimeError(f"Variable {name} found but missing bind")
					push(stack, ptr, val)
				case Const(val):
					push(stack, ptr, val)
				case UnaryExpr(op):
					op_fn = self.get_uop(op)
					unary(stack, ptr, op_fn)
				case BinaryExpr(op):
					op_fn = self.get_binop(op)
					binary(stack, ptr, op_fn)
				case _:
					raise RuntimeError(f"Unexpected node type: {type(node)}")
		if ptr != 1:
			raise RuntimeError(f"stack length is {ptr} at end of input. Should be 1")
		return stack[0]

	def infix(self) -> str:
		opstrings = {
				BinaryOp.ADD: "+",
				BinaryOp.SUB: "-",
				BinaryOp.MUL: "*",
				BinaryOp.INTDIV: "//",
				BinaryOp.MOD: "%",
				UnaryOp.ABS: "abs",
                UnaryOp.POW2: "pow2",
                UnaryOp.POW3: "pow3",
				UnaryOp.SIGN: "sign",
				UnaryOp.RELU: "relu",
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


if __name__ == "__main__":
	codes = [
		100, 'x0', 'pow2', 'x3', 'mul', 'mul', 200, 'x0', 'x1', 'x3', 'mul',
		'mul', 'mul', -50, 'x1', 'pow2', 'x3', 'mul', 'add', 'add', 'add'
    ]
	vals = [parse_rpn_value(c) for c in codes]
	expr = RPNExpression.from_vals(vals, 10)


