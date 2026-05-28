import numpy as np
import math
from typing import Union, Iterable, Self, Callable
from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
from enum import Enum
from dataclasses import dataclass
from .. import jfuncs
import random
import operator
from collections import Counter


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

class ControlOp(Enum):
	PLUS = "plus_sign"
	MINUS = "minus_sign"
	EQUALS = "equals_sign"

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
SymbolNode = Union[BinaryOp, UnaryOp, ControlOp, str]
MODULO_OPS = (
	BinaryOp.MOD_ADD,
	BinaryOp.MOD_SUB,
	BinaryOp.MOD_MUL,
	BinaryOp.MOD_INTDIV,
	UnaryOp.MOD_SQR
)

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
		self.mod_val = mod_val
		self.nodes = tuple(_postorder(node))
		self.depth = _depth(node) 


	def get_binop(self, op: BinaryOp) -> Callable[[int, int], int]:
		def _intdiv(x, y):
			mask = (y != 0)
			dest = np.zeros_like(y)
			dest[mask] = x[mask] // y[mask]
			return dest

		return {
				BinaryOp.ADD: lambda x, y: x + y,
				BinaryOp.SUB: lambda x, y: x - y,
				BinaryOp.MUL: lambda x, y: x * y,
				BinaryOp.INTDIV: lambda x, y: _intdiv(x, y),
				BinaryOp.MOD: lambda x, y: x % y,
				BinaryOp.MOD_ADD: lambda x, y: (x + y) % self.mod_val,
				BinaryOp.MOD_SUB: lambda x, y: (x - y) % self.mod_val,
				BinaryOp.MOD_MUL: lambda x, y: (x * y) % self.mod_val,
				BinaryOp.MOD_INTDIV: lambda x, y: _intdiv(x, y) % self.mod_val 
				}[op]
	
	def get_uop(self, op: UnaryOp) -> Callable[[int], int]:
		return {
				UnaryOp.ABS: abs,
				UnaryOp.SQR: lambda x: x ** 2,
				UnaryOp.SIGN: lambda x: np.where(x >= 0, 1, -1),
				UnaryOp.RELU: lambda x: np.maximum(0, x),
				UnaryOp.MOD_SQR: lambda x: x ** 2 % self.mod_val
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

	def __repr__(self):
		strs = []
		for v in self.token_vals:
			match v:
				case BinaryOp() | UnaryOp():
					strs.append(v.value)
				case str() | int():
					strs.append(str(v))
				case _:
					raise RuntimeError(f"Unexpected token val: {v}")
		return ' '.join(strs)

	@property
	def variables(self):
		return tuple(sorted(set(v.value for v in self.nodes if isinstance(v, Variable))))

	@property
	def const_values(self):
		return tuple(sorted(set(c.value for c in self.nodes if isinstance(c, Const))))

	@property
	def ops(self):
		return tuple(
				sorted(
					set(c.op for c in self.nodes if isinstance(c, (BinaryExpr, UnaryExpr))),
					key=lambda op: op.value 
					))

	@property
	def is_all_modulo(self):
		return all(op in MODULO_OPS for op in self.ops)

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

	def tokens_base_enc(
		self,
		op_map: dict[SymbolNode, int],
		use_dpse: bool,
		base: int,
		zero: int,
	) -> np.array:
		"""
		Convert expression to an int32 np.array of tokens.
		Integer values are base encoded as follows:

		op_map: mapping for SymbolNode to the integer token 
		base: the base to use for base encoding
		use_dpse: whether to use digit-place-specific encoding
		zero: the value for the digit zero

		if `use_dpse`, digits are encoded as:
		place0: [zero, zero + base)
		place1: [zero + base, zero + (base * 2))
		...

		otherwise, all digits are encoded as:
		[zero, zero + base)
		"""
		plus = op_map.get(ControlOp.PLUS)
		minus = op_map.get(ControlOp.MINUS)
		if plus is None or minus is None:
			raise RuntimeError("op_map lacks mapping for ControlOp.PLUS and/or MINUS")
		
		res = []
		for node in self.nodes:
			match node:
				case Variable(val):
					tok = op_map.get(val)
					if tok is None:
						raise RuntimeError(f"variable {val} not found in op_map")
					res.append(tok)
				case BinaryExpr() | UnaryExpr():
					tok = op_map.get(node)
					if tok is None:
						raise RuntimeError(f"op {node} not found in op_map")
					res.append(tok)
				case int(i) | Const(i):
				# case int(i):
					is_positive, digits = jfuncs.tokenize_one_int(i, base, use_dpse)
					if use_dpse:
						offsets = (np.arange(len(digits)) * base)[::-1]
						digits += offsets
					res.append(plus if is_positive else minus)
					res.extend((digits + zero).tolist())
				case _:
					raise RuntimeError(f"unrecognized node type: {type(node)}")
		return np.array(res)

	def tokens(
		self,
		op_map: dict[SymbolNode, int],
		zero: int,
	) -> np.array:
		res = []
		for node in self.nodes:
			match node:
				case Variable(val):
					tok = op_map.get(val)
					if tok is None:
						raise RuntimeError(f"variable value {val} not found in op_map")
					res.append(tok)
				case BinaryExpr(op) | UnaryExpr(op):
					tok = op_map.get(op)
					if tok is None:
						raise RuntimeError(f"variable value {val} not found in op_map")
					res.append(tok)
				case int(i) | Const(i):
					res.append(i + zero)
				case _:
					raise RuntimeError(f"unrecognized node type: {type(node)}")
		return np.array(res)

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
	"""
	An instance of this class theoretically defines the set of all expressions,
	represented as trees, in which leaf nodes can be one of `variables` or `consts`,
	and non-leaf nodes can be any of the `binops` (with two children) or `uops` (with
	one child).

	Once constructed, call gen_trees
	"""
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
	def n_vars(self):
		return len(self.variables)

	@property
	def n_consts(self):
		return len(self.consts)

	@property
	def n_binops(self):
		return len(self.binops)

	@property
	def n_uops(self):
		return len(self.uops)

	@property
	def leaf_vals(self):
		return self.variables + self.consts

	def get_counts(self, max_depth: int):
		"""
	    returns counts[d][(v, c)]: number of trees with v variables, c consts, and depth <= d
		"""
		# base cases
		leaf = { (1, 0): self.n_vars, (0, 1): self.n_consts }
		counts = [leaf]

		for _ in range(1, max_depth + 1):
			pre = counts[-1]
			cur = dict(leaf) 
			for k1, ct1 in pre.items():
				cur[k1] = cur.get(k1, 0) + ct1 * self.n_uops
				for k2, ct2 in pre.items():
					key = (k1[0] + k2[0], k1[1] + k2[1])
					cur[key] = cur.get(key, 0) + ct1 * ct2 * self.n_binops
			counts.append(cur)

		return counts

	def _unrank(self, index, v, c, d, counts) -> Node:
		"""
		Return tree with v variables, c consts, depth <= d corresponding to index
		"""
		if d == 0:
			# must return leaf
			if v == 0 and c == 1:
				assert index < self.n_consts, f"{index=} not in [0, {self.n_consts=})"
				return Const(self.consts[index])
			elif v == 1 and c == 0:
				assert index < self.n_vars, f"{index=} not in [0, {self.n_vars=})"
				return Variable(self.variables[index])
			else:
				raise RuntimeError(f"Invalid budget for depth 0")

		if v == 1 and c == 0:
			if index < self.n_vars:
				return Variable(self.variables[index])
			index -= self.n_vars
		if v == 0 and c == 1:
			if index < self.n_consts:
				return Const(self.consts[index])
			index -= self.n_consts

		prev_count = counts[d-1].get((v, c), 0)
		unary_total = self.n_uops * prev_count 
		if index < unary_total:
			op_idx, child_idx = divmod(index, prev_count)
			child = self._unrank(child_idx, v, c, d-1, counts)
			return UnaryExpr(self.uops[op_idx], child)
		index -= unary_total

		for v_l in range(v + 1):
			for c_l in range(c + 1):
				v_r, c_r = v - v_l, c - c_l
				num_left = counts[d-1].get((v_l, c_l), 0)
				num_right = counts[d-1].get((v_r, c_r), 0)
				prev_count = num_left * num_right
				cur_size = self.n_binops * prev_count 
				if index < cur_size:
					op_idx, tree_idx = divmod(index, prev_count)
					l_idx, r_idx = divmod(tree_idx, num_right)
					left = self._unrank(l_idx, v_l, c_l, d-1, counts)
					right = self._unrank(r_idx, v_r, c_r, d-1, counts)
					return BinaryExpr(self.binops[op_idx], left, right)
				index -= cur_size
		raise RuntimeError("index out of bounds.  search space exhausted.")

	def gen_trees(
		self, 
		seed: int, 
		num_vars: int,
		num_consts: int,
		max_depth: int, 
		max_trees: int
	) -> list[Node]:
		"""
		Generate at most `max_trees` trees of up to `max_depth` depth, containing
		`num_consts` constants and `num_vars` variables, sampled uniformly from all
		possible trees with these qualities.
		"""
		counts = self.get_counts(max_depth)
		total_trees = counts[-1].get((num_vars, num_consts))
		if total_trees is None:
			return []
		rng = random.Random(seed)
		max_trees = min(max_trees, total_trees)

		def _get_indices(total_trees, n):
			indices = set()
			while len(indices) < n:
				indices.add(rng.randrange(total_trees))
			return list(indices)

		indices = _get_indices(total_trees, max_trees)
		return [self._unrank(i, num_vars, num_consts, max_depth, counts) for i in indices]


def expr_cross_entropy(
	key: PRNGKeyArray,
	rpn: RPNExpression, 
	n_outputs: int, 
	n_trials: int, 
	input_range: list[int]
) -> float:
	"""
	Report averaage cross entropy between a uniform distribution
	over `n_output` values and the actual distribution
	"""
	def entropy(counts):
		probs = counts / counts.sum()
		ent = jnp.where(probs == 0, 0, probs * -jnp.log2(probs))
		return ent.sum()

	def hist(bins, inds):
		return jnp.zeros_like(bins).at[inds].add(1)

	V = len(rpn.variables)
	baseline_counts = jnp.unique(jnp.arange(n_outputs) % rpn.mod_val, return_counts=True)[1]
	baseline_ent = entropy(baseline_counts)

	B, C = n_trials, V + n_outputs
	vals_BC = jnp.empty((B, C), dtype=np.int32)
	vals_BC = vals_BC.at[:,:V].set(jax.random.choice(key, np.array(input_range), (B, V)))

	if rpn.is_all_modulo:
		max_bins = rpn.mod_val
	else:
		max_bins = B * C 

	for t in range(V, C):
		binds = { rpn.variables[vi]: vals_BC[:,t-vi-1] for vi in range(V) }
		vals_BC = vals_BC.at[:,t].set(rpn.evaluate(**binds))
	
	outs_BC = vals_BC[:,V:]
	bins = jnp.unique(outs_BC, size=max_bins, fill_value=-1)
	bins = jnp.sort(bins)
	inds_BC = jnp.searchsorted(bins, outs_BC)
	counts_BN = jax.vmap(hist, in_axes=(None, 0))(bins, inds_BC)
	ents_B = jax.vmap(entropy)(counts_BN)

	"""
	jax.debug.print(
			"baseline_counts:\n{}\n"
			"baseline_ent:\n{}\n"
			"bins:\n{}\n"
			"outs_BC:\n{}\n"
			"counts_BN:\n{}\n"
			"counts_BN.sum(axis=1):\n{}\n",
			baseline_counts, baseline_ent, bins, outs_BC, counts_BN, counts_BN.sum(axis=1)
	)
	"""
	return ents_B.mean() / baseline_ent 


def entropy_fraction(outputs_BC: jax.Array, max_bins: int) -> jax.Array:
	"""
	Computes the average fraction of maximum possible entropy that the `outputs_BC` 
	attain, assuming `max_bins` is the maximal possible value.
	"""
	def hist_fn(bins, inds):
		return jnp.zeros_like(bins).at[inds].add(1)

	B, C = outputs_BC.shape 
	baseline_counts = jnp.unique(jnp.arange(C) % max_bins, size=max_bins, return_counts=True)[1]
	baseline_ent = jfuncs.entropy(baseline_counts).sum()
	
	bins = jnp.unique(outputs_BC, size=max_bins, fill_value=-1)
	bins = jnp.sort(bins)
	inds_BC = jnp.searchsorted(bins, outputs_BC)
	counts_BN = jax.vmap(hist_fn, in_axes=(None, 0))(bins, inds_BC)
	ents_B = jax.vmap(jfuncs.entropy)(counts_BN).sum(axis=1)
	return jnp.where(baseline_ent == 0.0, 1.0, ents_B.mean() / baseline_ent)


