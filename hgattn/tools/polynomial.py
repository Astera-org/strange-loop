import itertools
from typing import Iterator
from dataclasses import dataclass
import numpy as np


@dataclass
class PolynomialOpts:
	total_vars: int
	min_terms: int
	max_terms: int
	min_arity: int
	max_arity: int
	min_degree: int
	max_degree: int
	min_const_coeff: int
	max_const_coeff: int
	min_coeff: int
	max_coeff: int
	int_base: int

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

def structures(a, d, t) -> Iterator[tuple[tuple[int]]]:
	"""
	Row-sets over slots 0..a-1 with max total degree exactly d and every slot used,
	and no more than t total terms. Rows descending; constant row not included.  Uses
	a backtracking approach
	"""
	if a == 0:
		if d == 0:
			yield ()
			return
	if d == 0:
		return
	U = monomials(a, d, min_deg=1)
	n, full = len(U), (1 << a) - 1
	supp = [sum(1 << i for i, e in enumerate(m) if e) for m in U]
	isdeg = [sum(m) == d for m in U]
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


class Polynomial:
	def __init__(
		self, 
		term_powers: tuple[tuple[int]],
		variables: tuple[str],
		coefficients: tuple[str],
		const_coeff: str|None,
	):
		self.term_powers = term_powers
		self.variables = variables
		self.coefficients = coefficients 
		self.const_coeff = const_coeff

	@property
	def input_span(self):
		return max((int(v[1:]) + 1 for v in self.variables), default=0)

	def __hash__(self):
		return hash((self.term_powers, self.variables, self.const_coeff))

	def __repr__(self):
		s = []

		for coeff, tp in zip(self.coefficients, self.term_powers):
			term = []
			for v, p in zip(self.variables, tp):
				if p == 0:
					continue
				elif p == 1:
					term.append(v)
				else:
					term.append(f"{v}^{p}")
			s.append(" ".join((coeff, *term)))
		if self.const_coeff is not None:
			s.append(self.const_coeff)
		return ' + '.join(s)

	def to_rpn(self) -> tuple[str]:
		res = []
		ops = []
		def _term_to_rpn(coeff, pows):
			res = [coeff]
			ops = []
			for v, p in zip(self.variables, pows):
				if p == 0:
					continue
				res.append(v)
				ops.append('MUL')
				if p > 1:
					res.append(f"POW{p}")
			return tuple(res + ops)

		for coeff, tp in zip(self.coefficients, self.term_powers):
			res.extend(_term_to_rpn(coeff, tp))
		ops.extend(['ADD'] * (len(self.term_powers) - 1))

		if self.const_coeff is not None:
			res.append(self.const_coeff)
			ops.append('ADD')

		return res + ops

	def to_rpn_code(self, codes: tuple[str], max_size: int) -> np.ndarray:
		"""
		Encode the rpn representation, right-padding to `max_size`
		"""
		code_map = { c: idx for idx, c in enumerate(codes) }
		noop = code_map.get('NOOP')
		if noop is None:
			raise RuntimeError(f"codes did not contain 'NOOP' code")
		buf = np.full(max_size, noop, dtype=np.uint32)
		rpn_vals = self.to_rpn()
		if len(rpn_vals) > max_size:
			raise RuntimeError(
				f"RPN expression is {len(rpn_vals)} values > max_size of {max_size}")

		for idx, val in enumerate(rpn_vals):
			tok = code_map.get(val)
			if tok is None:
				raise RuntimeError(f"codes did not contain `{tok}` token")
			buf[idx] = tok
		return buf



class PolyGen:
	"""
	A generator for Polynomials
	"""
	def __init__(self, opts: PolynomialOpts):
		self.opts = opts
		self.variables = tuple(f"v{i}" for i in range(opts.total_vars)) 
		self.coefficients = tuple(f"c{i}" for i in range(1, opts.max_terms + 1))
		self.const_coeff = "c0"

	@property
	def max_rpn_length(self):
		"""
		Return the maximum length of any Polynomial.to_rpn() result.
		This is conservative since it's hard to compute exactly.
		"""
		# 3 units for each variable: MUL, POW#, variable
		# 2 units for const term: ADD
		max_monomial = self.opts.max_arity * 3 - 1
		adds = self.opts.max_terms # max_terms - 1 for non-const terms, 1 for const
		return self.opts.max_terms * max_monomial + adds

	@property
	def max_infix_token_length(self):
		"""
		The maximal length of a tokenization of this formula represented in infix
		"""
		max_mag = max(abs(self.opts.min_coeff), abs(self.opts.max_coeff))
		max_coeff_tokens = 1 + jfuncs.get_max_digits(max_mag, self.opts.int_base)
		max_const_mag = max(abs(self.opts.min_const_coeff), abs(self.opts.max_const_coeff))
		max_const_coeff_tokens = 1 + jfuncs.get_max_digits(max_const_mag, self.opts.int_base)

		# adds are explicit, muls implicit
		coeff_tokens = max_int_tokens * self.opts.max_terms + max_const_coeff_tokens
		pow_var_tokens = self.opts.max_terms * self.opts.max_arity * 2 # POW and variable
		plus_tokens = self.opts.max_terms # between terms, plus 1 for const term
		return coeff_tokens + pow_var_tokens + plus_tokens

	@property
	def max_rpn_stack_depth(self):
		"""
		The maximum stack depth needed to evaluate any RPN expression
		"""
		return self.opts.max_terms + self.opts.max_arity + 1

	@property
	def codes(self) -> dict[str, int]:
		ops = 'NOOP', 'ADD', 'MUL', 'POW2', 'POW3'
		return tuple((*ops, *self.variables, self.const_coeff, *self.coefficients))

	def generate(self) -> Iterator[Polynomial]:
		"""
		Generate all possible polynomials within the constraints in opts
		"""
		for a in range(self.opts.min_arity, self.opts.max_arity + 1):
			for d in range(self.opts.min_degree, self.opts.max_degree + 1):
				for st in structures(a, d, self.opts.max_terms):
					for vs in itertools.combinations(self.variables, a):
						cs = self.coefficients[:len(vs)]
						for cc in (None, self.const_coeff):
							yield Polynomial(st, vs, cs, cc)

