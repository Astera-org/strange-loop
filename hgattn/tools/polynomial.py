import itertools
from typing import Iterator
from dataclasses import dataclass
import numpy as np
from .mathops import BinaryOp, UnaryOp
from .rpn import RPNExpression, parse_rpn_value


class PolyTemplate:
	def __init__(
		self,
		monomial_inds: np.array, # i32[t] = m, term t is PolyGen.monomials[m]
		term_count: int,
		degree: int,
		arity: int,
	):
		self.monomial_inds = monomial_inds
		self.term_count = term_count
		self.degree = degree
		self.arity = arity
		# i32[v] = p  variable v binds to state position s 
		# state positions go backwards: [5 4 3 2 1 0 *], where * denotes output of polynomial

	@property
	def input_span(self):
		return np.max(self.variable_inds[:self.arity]) + 1

	@property
	def coefficients(self):
		return tuple(f"c{t}" for t in range(self.term_count))

	def __hash__(self):
		return hash((self.monomial_inds, self.variable_inds))

	def to_rpn(self, monomials: np.ndarray) -> tuple[BinaryOp|UnaryOp|str]:
		res = []
		ops = []
		def _term_to_rpn(coeff, pows):
			res = [coeff]
			ops = []
			for vi, p in zip(self.variable_inds, pows):
				if p == 0:
					continue
				res.append(f"v{vi}")
				ops.append(BinaryOp.MUL)
				if p == 2: res.append(UnaryOp.POW2)
				elif p == 3: res.append(UnaryOp.POW3)
			return tuple(res + ops)

		for coeff, tp in zip(self.coefficients, monomials):
			res.extend(_term_to_rpn(coeff, tp))
		ops.extend([BinaryOp.ADD] * (len(self.term_powers) - 1))

		return res + ops

	def to_infix(self, monomials: np.ndarray) -> tuple[BinaryOp|UnaryOp|str]: 
		res = []
		for coeff, tp in zip(self.coefficients, monomials):
			res.append(coeff)
			for vi, p in zip(self.variables_inds, tp):
				if p == 0:
					continue
				res.append(f"v{vi}")
				if p == 1:
					pass
				elif p == 2: res.append(UnaryOp.POW2)
				elif p == 3: res.append(UnaryOp.POW3)
				else:
					raise RuntimeError(f"Powers above 3 are not supported")
		return tuple(res) 

	def __repr__(self):
		return "\n".join((f"{k}: {v}" for k, v in self.__dict__.items()))

	def _encode(
			self, 
			codes: tuple[str|BinaryOp|UnaryOp], 
			vals: tuple[str|BinaryOp|UnaryOp],
			max_size: int,
			) -> np.ndarray:
		"""
		Encode the rpn representation, right-padding to `max_size`
		"""
		code_map = { c: idx for idx, c in enumerate(codes) }
		noop = code_map.get('NOOP')
		if noop is None:
			raise RuntimeError(f"codes did not contain 'NOOP' code")
		buf = np.full(max_size, noop, dtype=np.uint32)
		if len(vals) > max_size:
			raise RuntimeError(
					f"RPN expression is {len(vals)} values > max_size of {max_size}")

		for idx, val in enumerate(vals):
			tok = code_map.get(val)
			if tok is None:
				raise RuntimeError(f"codes did not contain `{tok}` token")
			buf[idx] = tok
		return buf

	def to_rpn_code(self, codes: tuple[str|BinaryOp|UnaryOp], max_size: int) -> np.ndarray:
		rpn_vals = self.to_rpn()
		return self._encode(codes, rpn_vals, max_size)

	def to_infix_code(self, codes: tuple[str|BinaryOp|UnaryOp], max_size: int) -> np.ndarray:
		infix_vals = self.to_infix()
		return self._encode(codes, infix_vals, max_size)


class PolyGen:
	"""
	A generator for Polynomials
	"""
	def __init__(
			self, 
			total_vars: int,
			term_counts: list[int],
			arities: list[int],
			degrees: list[int],
	):
		self.total_vars = total_vars
		self.term_counts = tuple(term_counts)
		self.max_terms = max(term_counts)
		self.arities = tuple(arities)
		self.max_arity = max(arities)
		self.degrees = tuple(degrees)
		self.max_degree = max(degrees)
		self.variables = tuple(f"x{i}" for i in range(self.total_vars)) 
		self.coefficients = tuple(f"c{i}" for i in range(self.max_terms))

	def monomials(self) -> np.array:
		"""
		Generate the array of exponent vectors, one per row.
		Each has arity <= self.max_arity and degree <= self.max_degree
		"""
		def rec(prefix, slots_left, arity_left, degree_left):
			if slots_left == 0:
				yield tuple(prefix)
				return
			hi = degree_left if arity_left > 0 else 0
			for expon in range(hi + 1):
				prefix.append(expon)
				yield from rec(
						prefix, slots_left - 1, arity_left - (expon != 0), degree_left - expon)
				prefix.pop()

		out = list(rec([], self.total_vars, self.max_arity, self.max_degree))
		return np.array(out, dtype=np.int32)

	def templates(self) -> Iterator[PolyTemplate]:
		"""
		censored backtracking to enumerate all possible polynomial templates with
		given stats
		"""
		if len(self.arities) == 0 or len(self.degrees) == 0 or len(self.term_counts) == 0:
			return

		max_degree = max(self.degrees)
		U = self.monomials()
		n, full = len(U), (1 << self.max_arity) - 1

		# supp[i] is the i'th monomial's support mask over the unknowns
		supp = [sum(1 << i for i, e in enumerate(m) if e) for m in U]

		mon_degrees = [sum(m) for m in U]
		suf_supp = [0] * (n + 1)
		suf_degrees = [set()] * (n + 1)
		for i in range(n - 1, -1, -1):
			suf_supp[i] = suf_supp[i + 1] | supp[i]
			suf_degrees[i] = suf_degrees[i + 1].union({mon_degrees[i]})

		mon_used = np.full((n,), False)
		inds = np.full((n,), 0, dtype=np.int32)

		def rec(i, vars_used, degree):
			arity = vars_used.bit_count()
			if arity > self.max_arity:
				return

			if np.sum(mon_used) > self.max_terms:
				return

			if degree > max_degree:
				return

			if i == n:
				term_count = np.sum(mon_used)
				if (
					term_count in self.term_counts 
					and arity in self.arities 
					and degree in self.degrees
				):
					mon_inds = np.full((n,), 0, dtype=np.int32) 
					mon_inds[:term_count] = np.flatnonzero(mon_used)
					yield PolyTemplate(mon_inds, term_count, degree, arity)
				return

			mon_used[i] = True
			yield from rec(i + 1, vars_used | supp[i], max(degree, mon_degrees[i]))
			mon_used[i] = False
			yield from rec(i + 1, vars_used, degree)
		yield from rec(0, 0, 0)

	@property
	def max_rpn_length(self):
		"""
		Return the maximum length of any Polynomial.to_rpn() result.
		This is conservative since it's hard to compute exactly.
		"""
		# 3 units for each variable: MUL, POW#, variable
		# 2 units for `coeff ... mul` at the end 
		max_monomial = self.max_arity * 3 + 2
		adds = self.max_terms - 1 
		return self.max_terms * max_monomial + adds

	@property
	def max_infix_length(self):
		"""
		Return the maximum length of any Polynomial.to_infix_code() result.
		"""
		max_monomial = self.max_arity * 2 + 1
		return self.max_terms * max_monomial

	@property
	def max_rpn_stack_depth(self):
		"""
		The maximum stack depth needed to evaluate any RPN expression
		"""
		return self.max_terms + self.max_arity + 1

	@property
	def codes(self) -> tuple[BinaryOp|UnaryOp|str]:
		ops = 'NOOP', BinaryOp.ADD, BinaryOp.MUL, UnaryOp.POW2, UnaryOp.POW3
		return tuple((*ops, *self.variables, *self.coefficients))

	@property
	def code_map(self):
		return { code: idx for idx, code in enumerate(self.codes) }

if __name__ == "__main__":
	pg = PolyGen(total_vars=5, term_counts=(1,2,3,4), arities=(1,2,3), degrees=(1,2,3))
	monomials = pg.monomials()
	polys = list(pg.templates())
	inds = np.random.randint(low=0, high=len(polys), size=10) 

	print("Polynomial natural representations")
	for i in inds:
		print(str(polys[i]))

	print("\n")
	print("RPN representations")
	for i in inds:
		print(" ".join(polys[i].to_rpn(monomials)))

	print("Test all polynomials export valid RPN expressions")
	def subst_const(code):
		if isinstance(code, str) and code.startswith('c'):
			return 10
		return code


	for p in polys:
		# rpn_codes = p.to_rpn()
		# rpn_subst = [subst_const(co) for co in rpn_codes]
		# rpn_vals = [parse_rpn_value(co) for co in rpn_subst]
		# expr = RPNExpression.from_vals(rpn_vals, 2**32)

		infix = p.to_infix(monomials)
		infix_codes = p.to_infix_code(pg.codes, pg.max_infix_length)
		print(infix_codes)


