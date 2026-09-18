import itertools
from typing import Iterator
from dataclasses import dataclass
import numpy as np
from .mathops import BinaryOp, UnaryOp
from .rpn import RPNExpression, parse_rpn_value


class Polynomial:
	def __init__(
		self, 
		term_powers: tuple[tuple[int]],
		variables: tuple[str],
		coefficients: tuple[str],
	):
		if len(term_powers) != len(coefficients):
			raise RuntimeError(f"{len(term_powers)=} != {len(coefficients)=}") 
		self.term_powers = term_powers
		self.variables = variables
		self.coefficients = coefficients 

	@property
	def input_span(self):
		return max((int(v[1:]) + 1 for v in self.variables), default=0)

	def __hash__(self):
		return hash((self.term_powers, self.variables))

	def to_rpn(self) -> tuple[BinaryOp|UnaryOp|str]:
		res = []
		ops = []
		def _term_to_rpn(coeff, pows):
			res = [coeff]
			ops = []
			for v, p in zip(self.variables, pows):
				if p == 0:
					continue
				res.append(v)
				ops.append(BinaryOp.MUL)
				if p == 2: res.append(UnaryOp.POW2)
				elif p == 3: res.append(UnaryOp.POW3)
			return tuple(res + ops)

		for coeff, tp in zip(self.coefficients, self.term_powers):
			res.extend(_term_to_rpn(coeff, tp))
		ops.extend([BinaryOp.ADD] * (len(self.term_powers) - 1))

		return res + ops

	def to_infix(self) -> tuple[BinaryOp|UnaryOp|str]: 
		res = []
		for coeff, tp in zip(self.coefficients, self.term_powers):
			res.append(coeff)
			for v, p in zip(self.variables, tp):
				if p == 0:
					continue
				res.append(v)
				if p == 1:
					pass
				elif p == 2: res.append(UnaryOp.POW2)
				elif p == 3: res.append(UnaryOp.POW3)
				else:
					raise RuntimeError(f"Powers above 3 are not supported")
		return tuple(res) 

	def __repr__(self):
		infix = self.to_infix()
		terms = [str(term) for term in infix]
		return " ".join(terms)

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


def monomials(a, max_deg, min_deg=0):
	"""
	Exponent vectors with `a` slots with min_deg <= total <= max_deg, emitted in
	descending lexicographic order.
	"""
	if a == 0:
		return [()] if min_deg == 0 else []
	out = []
	def rec(prefix, slots, budget):
		if slots == 0:
			if min_deg <= sum(prefix) <= max_deg:
				out.append(tuple(prefix))
			return
		for e in range(budget, -1, -1):
			prefix.append(e)
			rec(prefix, slots - 1, budget - e)
			prefix.pop()
	rec([], a, max_deg)
	return out

@dataclass
class PolyTemplate:
	monomials: tuple[tuple[int]]
	degree: int
	arity: int


def structures(
	term_counts: list[int],
	arities: list[int],
	degrees: list[int],
) -> Iterator[PolyTemplate]:
	"""
	censored backtracking to enumerate all possible polynomial templates with
	given stats
	"""
	if len(arities) == 0 or len(degrees) == 0 or len(term_counts) == 0:
		return

	max_term_count = max(term_counts)
	max_arity = max(arities)
	max_degree = max(degrees)
	U = monomials(max_arity, max_degree, min_deg=1)
	n, full = len(U), (1 << max_arity) - 1

	# supp[i] is the i'th monomial's support mask over the unknowns
	supp = [sum(1 << i for i, e in enumerate(m) if e) for m in U]

	mon_degrees = [sum(m) for m in U]
	suf_supp = [0] * (n + 1)
	suf_degrees = [set()] * (n + 1)
	for i in range(n - 1, -1, -1):
		suf_supp[i] = suf_supp[i + 1] | supp[i]
		suf_degrees[i] = suf_degrees[i + 1].union({mon_degrees[i]})

	chosen = []
	def rec(i, vars_used, degree):
		arity = vars_used.bit_count()
		if arity > max_arity:
			return
		# if vars_used | suf_supp[i] != full:
		# 	return # some slot can no longer be covered

		# if len(suf_degrees[i].intersection({degree})) == 0:
		# 	return # degree d can no longer be reached

		if len(chosen) > max_term_count:
			return

		if degree > max_degree:
			return

		if i == n:
			if (len(chosen) in term_counts 
			    and arity in arities 
			    and degree in degrees):
				yield PolyTemplate(tuple(chosen), degree, arity)
			return
		chosen.append(U[i])
		yield from rec(i + 1, vars_used | supp[i], max(degree, mon_degrees[i]))
		chosen.pop()
		yield from rec(i + 1, vars_used, degree)
	yield from rec(0, 0, 0)


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
		self.variables = tuple(f"x{i}" for i in range(self.total_vars)) 
		self.coefficients = tuple(f"c{i}" for i in range(self.max_terms))

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

	def generate(self) -> Iterator[Polynomial]:
		"""
		Generate all possible polynomials within the constraints
		"""
		for tmpl in structures(self.term_counts, self.arities, self.degrees):
			for vs in itertools.combinations(self.variables, tmpl.arity):
				cs = self.coefficients[:len(tmpl.monomials)]
				yield Polynomial(term_powers=tmpl.monomials, variables=vs, coefficients=cs)

if __name__ == "__main__":
	pg = PolyGen(total_vars=5, term_counts=(1,2,3,4), arities=(1,2,3), degrees=(1,2,3))
	polys = list(pg.generate())
	inds = np.random.randint(low=0, high=len(polys), size=10) 

	print("Polynomial natural representations")
	for i in inds:
		print(str(polys[i]))

	print("\n")
	print("RPN representations")
	for i in inds:
		print(" ".join(polys[i].to_rpn()))

	print("Test all polynomials export valid RPN expressions")
	def subst_const(code):
		if isinstance(code, str) and code.startswith('c'):
			return 10
		return code

	for p in polys:
		rpn_codes = p.to_rpn()
		rpn_subst = [subst_const(co) for co in rpn_codes]
		rpn_vals = [parse_rpn_value(co) for co in rpn_subst]
		expr = RPNExpression.from_vals(rpn_vals, 2**32)

		infix = p.to_infix()
		infix_codes = p.to_infix_code(pg.codes, pg.max_infix_length)
		print(infix)


