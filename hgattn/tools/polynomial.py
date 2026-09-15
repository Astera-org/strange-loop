import itertools
from typing import Iterator
from dataclasses import dataclass
import numpy as np
from .mathops import BinaryOp, UnaryOp
from .rpn import RPNExpression, parse_rpn_value


@dataclass
class PolynomialOpts:
	total_vars: int  # total number of distinct variables that can be used 
	min_terms: int   # non-constant monomial terms constrained to [min_terms, max_terms]
	max_terms: int
	min_arity: int   # (number of distinct variables with non-zero exponent in any one monomial)
	max_arity: int
	min_degree: int  # (maximum of sum of exponents of any monomial)
	max_degree: int
	min_const_coeff: int # range to sample the const coefficient
	max_const_coeff: int
	min_coeff: int       # range to sample the non-const coefficient
	max_coeff: int

	@property
	def max_int_magnitude(self):
		return max(
				abs(self.min_const_coeff), abs(self.max_const_coeff),
				abs(self.min_coeff), abs(self.max_coeff))


class Polynomial:
	def __init__(
		self, 
		term_powers: tuple[tuple[int]],
		variables: tuple[str],
		coefficients: tuple[str],
		const_coeff: str|None,
	):
		if len(term_powers) != len(coefficients):
			raise RuntimeError(f"{len(term_powers)=} != {len(coefficients)=}") 
		self.term_powers = term_powers
		self.variables = variables
		self.coefficients = coefficients 
		self.const_coeff = const_coeff

	@property
	def input_span(self):
		return max((int(v[1:]) + 1 for v in self.variables), default=0)

	def __hash__(self):
		return hash((self.term_powers, self.variables, self.const_coeff))

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

		if self.const_coeff is not None:
			res.append(self.const_coeff)
			ops.append(BinaryOp.ADD)

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

		if self.const_coeff is not None:
			res.append(self.const_coeff)
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

def structures(arity, deg, t) -> Iterator[tuple[tuple[int]]]:
	"""
	Row-sets over slots 0..arity-1 with max total degree exactly `deg` and every slot used,
	and no more than t total terms. Rows descending; constant row not included.  Uses
	a backtracking approach
	"""
	if arity == 0:
		if deg == 0:
			yield ()
			return
	if deg == 0:
		return
	U = monomials(arity, deg, min_deg=1)
	n, full = len(U), (1 << arity) - 1
	supp = [sum(1 << i for i, e in enumerate(m) if e) for m in U]
	isdeg = [sum(m) == deg for m in U]
	suf_supp = [0] * (n + 1)
	suf_deg = [False] * (n + 1)
	for i in range(n - 1, -1, -1):
		suf_supp[i] = suf_supp[i + 1] | supp[i]
		suf_deg[i] = suf_deg[i + 1] or isdeg[i]

	chosen = []
	def rec(i, mask, hit):
		if mask | suf_supp[i] != full: return # some slot can no longer be covered
		if not hit and not suf_deg[i]: return # degree d can no longer be reached
		if len(chosen) > t:
			return
		if i == n:
			yield tuple(chosen)
			return
		chosen.append(U[i])
		yield from rec(i + 1, mask | supp[i], hit or isdeg[i])
		chosen.pop()
		yield from rec(i + 1, mask, hit)
	yield from rec(0, 0, False)


class PolyGen:
	"""
	A generator for Polynomials
	"""
	def __init__(self, opts: PolynomialOpts):
		self.opts = opts
		self.variables = tuple(f"x{i}" for i in range(opts.total_vars)) 
		self.coefficients = tuple(f"c{i}" for i in range(1, opts.max_terms + 1))
		self.const_coeff = "c0"

	@property
	def max_rpn_length(self):
		"""
		Return the maximum length of any Polynomial.to_rpn() result.
		This is conservative since it's hard to compute exactly.
		"""
		# 3 units for each variable: MUL, POW#, variable
		# 2 units for `coeff ... mul` at the end 
		max_monomial = self.opts.max_arity * 3 + 2
		adds = self.opts.max_terms # max_terms - 1 for non-const terms, 1 for const
		return self.opts.max_terms * max_monomial + adds

	@property
	def max_infix_length(self):
		"""
		Return the maximum length of any Polynomial.to_infix_code() result.
		"""
		max_monomial = self.opts.max_arity * 2 + 1
		return self.opts.max_terms * max_monomial + 1 # + 1 for final const coeff


	@property
	def max_rpn_stack_depth(self):
		"""
		The maximum stack depth needed to evaluate any RPN expression
		"""
		return self.opts.max_terms + self.opts.max_arity + 1

	@property
	def codes(self) -> tuple[BinaryOp|UnaryOp|str]:
		ops = 'NOOP', BinaryOp.ADD, BinaryOp.MUL, UnaryOp.POW2, UnaryOp.POW3
		return tuple((*ops, *self.variables, self.const_coeff, *self.coefficients))

	@property
	def code_map(self):
		return { code: idx for idx, code in enumerate(self.codes) }

	def generate(self) -> Iterator[Polynomial]:
		"""
		Generate all possible polynomials within the constraints in opts
		"""
		for arity in range(self.opts.min_arity, self.opts.max_arity + 1):
			for deg in range(self.opts.min_degree, self.opts.max_degree + 1):
				for st in structures(arity, deg, self.opts.max_terms):
					for vs in itertools.combinations(self.variables, arity):
						cs = self.coefficients[:len(st)]
						for cc in (None, self.const_coeff):
							yield Polynomial(
								term_powers=st, 
								variables=vs, 
								coefficients=cs, 
								const_coeff=cc
							)

if __name__ == "__main__":
	opts = PolynomialOpts(
			total_vars=5,
			min_terms=1,
			max_terms=4,
			min_arity=1,
			max_arity=3,
			min_degree=1,
			max_degree=3,
			min_const_coeff=-10,
			max_const_coeff=10,
			min_coeff=-1000,
			max_coeff=1000
	)
	pg = PolyGen(opts)
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
		print(infix_codes)


