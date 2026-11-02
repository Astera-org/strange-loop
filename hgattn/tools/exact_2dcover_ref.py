import fire
import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from jaxtyping import Int, Int8, Array, Bool, PRNGKeyArray
from typing import NamedTuple, Iterable


class State:
	IN = 0
	OUT = 1
	UND = 2

	def __init__(self, R: int, W: int):
		self.R = np.full((R,), self.UND)
		self.W = np.full((W,), self.UND)
		self.level = np.int32(0)

	def __repr__(self):
		stats = dict(num_inR=self.in_R.sum(),
			   num_outR = self.out_R.sum(),
			   num_inW = self.in_W.sum(),
			   num_outW = self.out_W.sum(),
			   num_undR=self.und_R.sum(),
			   num_undW=self.und_W.sum(),
			   level=self.level)
		return " ".join(f"{k}: {v.item()}" for k, v in stats.items())

	@property
	def in_R(self):
		return self.R == self.IN

	@property
	def out_R(self):
		return self.R == self.OUT

	@property
	def in_W(self):
		return self.W == self.IN

	@property
	def out_W(self):
		return self.W == self.OUT

	@property
	def und_R(self):
		return self.R == self.UND

	@property
	def und_W(self):
		return self.W == self.UND

	def apply(self, in_R, out_R, in_W, out_W):
		self.R[in_R] = self.IN
		self.R[out_R] = self.OUT
		self.W[in_W] = self.IN
		self.W[out_W] = self.OUT
		return self

	def spawn(self) -> 'State':
		st = State(0, 0)
		st.R = self.R.copy()
		st.W = self.W.copy()
		st.level = self.level + 1
		return st



def gen_graph(R: int, W: int, C: int, n_witness_per_link: int, rng: Generator) -> np.array:
	graph = np.zeros((R, W, C), dtype=np.int32)
	rand_vals = rng.random((R, W, C))
	topk_RWC = np.argpartition(rand_vals, n_witness_per_link, axis=1)
	k_inds = topk_RWC[:,:n_witness_per_link,:]
	graph[np.arange(R)[:,None,None], k_inds, np.arange(C)[None,None,:]] = 1
	return graph

def gen_factored_graph(
		R: int, 
		W: int, 
		C: int, 
		witness_per_row: int,
		witness_per_col: int,
		rng: Generator
		) -> np.array:
	"""
	Generate a factored graph expressed as:
	fac1_RW[:,:,None] * fac2_WC[None,:,:]
	"""
	fac1_RW = np.zeros((R, W), dtype=np.int32)
	fac2_WC = np.zeros((W, C), dtype=np.int32)
	rand1 = rng.random((R, W))
	inds_RW = np.argpartition(rand1, witness_per_row, axis=1)
	inds_RK = inds_RW[:,:witness_per_row]

	rand2 = rng.random((W, C))
	inds_WC = np.argpartition(rand2, witness_per_col, axis=0)
	inds_KC = inds_WC[:witness_per_col,:]

	fac1_RW[np.arange(R)[:,None],inds_RK] = 1
	fac2_WC[inds_KC,np.arange(C)[None,:]] = 1

	return fac1_RW[:,:,None] * fac2_WC[None,:,:]


class ExactCover2D:
	def __init__(
			self, 
			graph_RWC: np.ndarray, 
			budgetR: int, 
			budgetW: int,
			max_depth: int,
			rng: Generator,
			):
		self.graph_RWC = graph_RWC
		self.budgetR = budgetR
		self.budgetW = budgetW
		self.max_depth = max_depth
		self.rng = rng 


	def propagate(self, st: State) -> State|None:
		"""
		advance the state, returning None if no possible solutions
		"""
		while True:
			sure_RC = (self.graph_RWC & st.in_W[None,:,None]).any(1)
			poss_RC = (self.graph_RWC & ~st.out_W[None,:,None]).any(1)
			cover_C = (st.in_R[:,None] & sure_RC).sum(0)
			cand_C = (~st.out_R[:,None] & poss_RC).sum(0)
			if (cover_C >= 2).any() or (cand_C == 0).any():
				return None
			covered_C = (cover_C == 1)

			out_R = st.und_R & (sure_RC & covered_C[None,:]).any(1)

			# witnesses whose addition would double-cover some column
			newcov_WC = (st.in_R[:,None,None] & (sure_RC[:,None,:] | self.graph_RWC)).sum(0)
			out_W = st.und_W & (newcov_WC >= 2).any(1)

			in_R, in_W = self.unit_rules(st, poss_RC, cand_C, covered_C)

			if (in_R & out_R).any() or (in_W & out_W).any():
				return None
			if not (out_R | in_R).any() and not (out_W | in_W).any():
				return st

			st = st.apply(in_R, out_R, in_W, out_W)

	def unit_rules(self, st: State, poss_RC, cand_C, covered_C) -> tuple[np.array, np.array]:
		open_C = ~covered_C

		# Candidate rows for each uncovered column: not OUT, with a possible edge.
		cand_RC = ~st.out_R[:,None] & poss_RC & open_C[None,:]

		# Row unit: an uncovered column with exactly one candidate row forces that row IN
		unit_C = open_C & (cand_C == 1)
		in_R = st.und_R & (cand_RC & unit_C[None,:]).any(1)

		# Witness unit: if every candidate path into an uncovered column c
		# goes through the same single non-OUT witness, that witness is forced IN
		path_WC = (cand_RC[:,None,:] & self.graph_RWC & ~st.out_W[None,:,None]).any(0)
		nW_C = path_WC.sum(0)
		unitW_C = open_C & (nW_C == 1)
		in_W = st.und_W & (path_WC & unitW_C[None,:]).any(1)

		return in_R, in_W

	def column_children(self, st: State, c: int) -> Iterable[State]:
		"""
		Children partitioning Ext(st) ixn S by (row covering c, first IN witness for it)
		"""
		graph_RW = self.graph_RWC[:,:,c]
		children = []
		for r in np.flatnonzero(st.R != State.OUT):
			Wrc = np.flatnonzero(graph_RW[r] & (st.W != State.OUT))
			if Wrc.size == 0:
				continue
			Wrc = self.rng.permutation(Wrc)

			if (st.W[Wrc] == State.IN).any():
				assert st.R[r] == State.UND
				ch = st.spawn()
				ch.R[r] = State.IN
				children.append(ch)
				continue

			for k, w in enumerate(Wrc):
				ch = st.spawn()
				assert st.R[r] != State.OUT
				assert st.W[w] == State.UND
				ch.R[r] = State.IN
				ch.W[w] = State.IN
				ch.W[Wrc[:k]] = State.OUT
				children.append(ch)

		return children


	def var_children(self, st: State) -> tuple[State, State]:
		"""
		Binary split on a random R or W UND variable into IN and OUT.
		This allows adding an additional row or witness without the need to cover an
		additional column.
		"""
		indsR = np.flatnonzero(st.und_R)
		indsW = np.flatnonzero(st.und_W)
		nr, nw = st.und_R.sum(), st.und_W.sum()
		idx = self.rng.choice(nr + nw)
		ch_in, ch_out = st.spawn(), st.spawn()

		if idx < nr:
			r = indsR[idx]
			assert st.R[r] == State.UND
			ch_in.R[r] = State.IN
			ch_out.R[r] = State.OUT
		else:
			w = indsW[idx - nr]
			assert st.W[w] == State.UND
			ch_in.W[w] = State.IN
			ch_out.W[w] = State.OUT
		return ch_in, ch_out

	def pick_uncovered_column(self, st: State) -> int|None:
		"""
		Uncovered column with the fewest column_children().  Return None if no
		uncovered columns.
		"""
		cand_R = (st.R != State.OUT)
		avail_W = (st.W != State.OUT)

		sure_RC = (self.graph_RWC & st.in_W[None,:,None]).any(1)
		covered_C = (st.in_R[:,None] & sure_RC).any(0)
		if covered_C.all():
			return None

		nW_RC = (self.graph_RWC & avail_W[None,:,None]).sum(1)
		per_RC = np.where(sure_RC, 1, nW_RC) * cand_R[:,None]
		nchild_C = np.where(covered_C, np.iinfo(np.int64).max, per_RC.sum(0))

		best = np.flatnonzero(nchild_C == nchild_C.min())
		return best[self.rng.integers(len(best))]


	def complete(self, st: State, depth: int) -> bool:
		return depth >= self.max_depth or self.at_budget(st) or self.all_determined(st)

	def all_determined(self, st: State) -> bool:
		return np.all(st.R != State.UND) and np.all(st.W != State.UND)

	def at_budget(self, st: State) -> bool:
		return st.in_R.sum() == self.budgetR and st.in_W.sum() == self.budgetW

	def over_budget(self, st):
		return st.in_R.sum() > self.budgetR or st.in_W.sum() > self.budgetW

	def order(self, children):
		return [children[i] for i in self.rng.permutation(len(children))]

	def search(self, st: State, depth) -> State|None:
		self.search_iters += 1
		st = self.propagate(st)
		if st is None or self.over_budget(st):
			return None
		if self.complete(st, depth):
			return st
		c = self.pick_uncovered_column(st)
		if c is not None:
			children = self.column_children(st, c)
		else:
			children = self.var_children(st)
		for child in self.order(children):
			res = self.search(child, depth+1)
			if res is not None:
				return res

	def solve(self) -> State|None:
		self.search_iters = 0
		R, W, _ = self.graph_RWC.shape
		st = State(R, W)
		return self.search(st, 0), self.search_iters

	def validate(self, st: State) -> bool:
		marg_RC = self.graph_RWC[*np.ix_(st.in_R, st.in_W),:].any(1)
		return bool((marg_RC.sum(0) == 1).all())


def main(
	R: int, 
	W: int, 
	C: int,
	w_per_row: int,
	w_per_col: int,
	w_per_link: int,
	seed: int,
	budget_r: int,
	budget_w: int,
	max_depth: int,
	use_factored: bool,
):
	np.set_printoptions(linewidth=180, threshold=10000000)
	rng = np.random.default_rng(seed)
	if use_factored:
		graph_RWC = gen_factored_graph(R, W, C, w_per_row, w_per_col, rng)
	else:
		graph_RWC = gen_graph(R, W, C, w_per_link, rng)
	cover2D = ExactCover2D(graph_RWC, budget_r, budget_w, max_depth, rng)

	print(f"Graph Density: {(graph_RWC.sum() / graph_RWC.size):5.3f}")

	total = 100
	num_valid = 0
	for i in range(total):
		st, iters = cover2D.solve()
		is_valid = cover2D.validate(st)
		num_valid += int(is_valid)
		print(f"{i}: {is_valid=}, {st}, {iters=}")

	print(f"Found {num_valid=} out of {total}")


if __name__ == "__main__":
	fire.Fire(main)

