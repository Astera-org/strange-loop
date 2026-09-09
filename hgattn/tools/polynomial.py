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
		has_const: bool,
	):
		self.term_powers = term_powers
		self.variables = variables
		self.has_const = has_const

	def __hash__(self):
		return hash((self.term_powers, self.variables, self.has_const))

	def __repr__(self):
		s = []
		for tp in self.term_powers:
			term = ''
			for v, p in zip(self.variables, tp):
				if p == 0:
					continue
				elif p == 1:
					term += v
				else:
					term += f"{v}^{p}"
			s.append(term)
		if self.has_const:
			s.append('Const')
		return ' + '.join(s)


class PolyGen:
	"""
	A generator for Polynomials
	"""
	def __init__(self, opts: PolynomialOpts):
		self.opts = opts
		self.variables = tuple(chr(ord('A') + i) for i in range(opts.total_vars)) 

	def generate(self) -> Iterator[tuple[int]]:
		for a in range(self.opts.min_arity, self.opts.max_arity + 1):
			for d in range(self.opts.min_degree, self.opts.max_degree + 1):
				for st in structures(a, d, self.opts.max_terms):
					for vs in itertools.combinations(self.variables, a):
						for has_const in (True, False):
							yield Polynomial(st, vs, has_const)

