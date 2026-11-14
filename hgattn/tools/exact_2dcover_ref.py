import fire
import numpy as np
from numpy.random import Generator
from jaxtyping import Int, Int8, Array, Bool, PRNGKeyArray
from typing import NamedTuple, Iterable
from enum import Enum, auto
from dataclasses import dataclass
from .graph import gen_witness_graph, gen_factored_witness_graph

IN = 0
OUT = 1
UND = 2

"""
Reference implementation of Exact Cover with Witnesses.  Given an arbitrary (0, 1)
tensor graph_RWC, find a combination of mask_R and mask_W such that if:

marg_RC = graph_RWC[np.ix_(mask_R, mask_W),:].any(1)

then marg_RC is an exact cover of the columns.  That is, np.all(marg_RC.sum(0) == 1)
"""

class Assignment:

	def __init__(self, R: int, W: int):
		self.R = np.full((R,), UND)
		self.W = np.full((W,), UND)
		self.level = np.int32(0)
		self.extraR = np.int32(0)
		self.extraW = np.int32(0)

	def __repr__(self):
		stats = dict(
			num_inR=self.in_R.sum(),
			num_outR = self.out_R.sum(),
			num_inW = self.in_W.sum(),
			num_outW = self.out_W.sum(),
			num_undR=self.und_R.sum(),
			num_undW=self.und_W.sum(),
			extraR=self.extraR,
			extraW=self.extraW,
			level=self.level)
		return " ".join(f"{k}: {v.item()}" for k, v in stats.items())

	@property
	def in_R(self):
		return self.R == IN

	@property
	def out_R(self):
		return self.R == OUT

	@property
	def in_W(self):
		return self.W == IN

	@property
	def out_W(self):
		return self.W == OUT

	@property
	def und_R(self):
		return self.R == UND

	@property
	def und_W(self):
		return self.W == UND

	def apply(self, in_R, out_R, in_W, out_W):
		self.R[in_R] = IN
		self.R[out_R] = OUT
		self.W[in_W] = IN
		self.W[out_W] = OUT
		return self

	def spawn(self) -> 'Assignment':
		asg = Assignment(0, 0)
		asg.R = self.R.copy()
		asg.W = self.W.copy()
		asg.level = self.level + 1
		asg.extraR = self.extraR
		asg.extraW = self.extraW
		return asg

# Status of the Subtree Ext(S)
class Status(Enum):
	RUNNING = auto()
	SOLVED = auto()
	INFEASIBLE = auto()

@dataclass
class Config:
	"""
	Configures the search as a whole
	"""
	status: Status
	resume_backtrack: bool

@dataclass
class Frame:
	asg: Assignment | None = None
	perm: np.ndarray | None = None
	child_idx: int = 0
	column: int | None = None  # which column chosen for column children
	isRchild: bool = False
	var_idx: int | None = None # index into R ++ W chosen for var children

class Result(NamedTuple):
	status: Status
	asg: Assignment | None


class ExactCover2DRef:
	def __init__(
			self, 
			graph_RWC: np.ndarray, 
			num_extra_rows: int,
			num_extra_witnesses: int,
			max_backtrack_steps: int,
			max_propagate_steps: int,
			max_search_iters: int,
			rng: Generator,
			):
		self.graph_RWC = graph_RWC
		C = graph_RWC.shape[2]
		self.num_extraR = num_extra_rows
		self.num_extraW = num_extra_witnesses
		self.max_backtrack_steps = max_backtrack_steps
		self.max_propagate_steps = max_propagate_steps
		self.max_search_iters = max_search_iters
		self.rng = rng 


	def propagate(self, asg: Assignment) -> Assignment|None:
		"""
		advance the state, returning None if no possible solutions
		iterations of propagate are not necessary for correctness, just
		an optimization.
		"""
		for _ in range(self.max_propagate_steps):
			sure_RC = (self.graph_RWC & asg.in_W[None,:,None]).any(1)
			poss_RC = (self.graph_RWC & ~asg.out_W[None,:,None]).any(1)
			cover_C = (asg.in_R[:,None] & sure_RC).sum(0)
			cand_C = (~asg.out_R[:,None] & poss_RC).sum(0)

			# check if exact cover condition is violated or we're out of options
			if (cover_C >= 2).any() or (cand_C == 0).any():
				return None

			covered_C = (cover_C == 1)

			out_R = asg.und_R & (sure_RC & covered_C[None,:]).any(1)

			# witnesses whose addition would double-cover some column
			newcov_WC = (asg.in_R[:,None,None] & (sure_RC[:,None,:] | self.graph_RWC)).sum(0)
			out_W = asg.und_W & (newcov_WC >= 2).any(1)

			in_R, in_W = self.unit_rules(asg, poss_RC, cand_C, covered_C)

			if (in_R & out_R).any() or (in_W & out_W).any():
				return None
			if not (out_R | in_R).any() and not (out_W | in_W).any():
				return asg

			asg = asg.apply(in_R, out_R, in_W, out_W)
		return asg

	def unit_rules(self, asg: Assignment, poss_RC, cand_C, covered_C) -> tuple[np.array, np.array]:
		open_C = ~covered_C

		# Candidate rows for each uncovered column: not OUT, with a possible edge.
		cand_RC = ~asg.out_R[:,None] & poss_RC & open_C[None,:]

		# Row unit: an uncovered column with exactly one candidate row forces that row IN
		unit_C = open_C & (cand_C == 1)
		in_R = asg.und_R & (cand_RC & unit_C[None,:]).any(1)

		# Witness unit: if every candidate path into an uncovered column c
		# goes through the same single non-OUT witness, that witness is forced IN
		path_WC = (cand_RC[:,None,:] & self.graph_RWC & ~asg.out_W[None,:,None]).any(0)
		nW_C = path_WC.sum(0)
		unitW_C = open_C & (nW_C == 1)
		in_W = asg.und_W & (path_WC & unitW_C[None,:]).any(1)

		return in_R, in_W

	def column_children(self, asg: Assignment, c: int) -> Iterable[Assignment]:
		"""
		Children partitioning Ext(asg) ixn S by (row covering c, first IN witness for it)
		"""
		graph_RW = self.graph_RWC[:,:,c]
		children = []
		for r in np.flatnonzero(asg.R != OUT):
			Wrc = np.flatnonzero(graph_RW[r] & (asg.W != OUT))
			if Wrc.size == 0:
				continue

			if (asg.W[Wrc] == IN).any():
				assert asg.R[r] == UND
				ch = asg.spawn()
				ch.R[r] = IN
				children.append(ch)
				continue

			for k, w in enumerate(Wrc):
				ch = asg.spawn()
				assert asg.R[r] != OUT
				assert asg.W[w] == UND
				ch.R[r] = IN
				ch.W[w] = IN
				ch.W[Wrc[:k]] = OUT
				children.append(ch)

		return children

	def column_child(self, asg: Assignment, c: int, ch_idx: int) -> Assignment:
		children = self.column_children(asg, c)
		return children[ch_idx]

	def var_index(self, asg: Assignment) -> tuple[bool, int]:
		"""
		Choose a particular R or W for the UND variable.
		Returns:
		isR: bool
		index: index of R if isR else W
		"""
		nr, nw = asg.und_R.sum(), asg.und_W.sum()
		rwidx = self.rng.choice(nr + nw)
		isR = (rwidx < nr)
		idx = rwidx if isR else rwidx - nr
		return isR, idx

	def var_children(
		self, asg: Assignment, isRchild: bool, var_idx: int
	) -> tuple[Assignment, Assignment]:
		"""
		Binary split on a random R or W UND variable into IN and OUT.
		This allows adding an additional row or witness without the need to cover an
		additional column.
		"""
		indsR = np.flatnonzero(asg.und_R)
		indsW = np.flatnonzero(asg.und_W)
		ch_in, ch_out = asg.spawn(), asg.spawn()
		# ch_in = asg.spawn()

		if isRchild:
			r = indsR[var_idx]
			assert asg.R[r] == UND
			ch_in.R[r] = IN
			ch_out.R[r] = OUT
			ch_in.extraR = asg.extraR + 1
		else:
			w = indsW[var_idx]
			assert asg.W[w] == UND
			ch_in.W[w] = IN
			ch_out.W[w] = OUT
			ch_in.extraW = asg.extraW + 1

		# if ch_in.extraR > self.num_extraR or ch_in.extraW > self.num_extraW:
		# 	return (ch_out,)
		return ch_in, ch_out

	def var_child(self, asg: Assignment, isRchild: bool, var_idx: int, idx: int) -> Assignment:
		"""
		Choose one var child.  
		"""
		return self.var_children(asg, isRchild, var_idx)[idx]


	def pick_uncovered_column(self, asg: Assignment) -> int|None:
		"""
		Uncovered column with the fewest column_children().  Return None if no
		uncovered columns.
		"""
		cand_R = (asg.R != OUT)
		avail_W = (asg.W != OUT)

		sure_RC = (self.graph_RWC & asg.in_W[None,:,None]).any(1)
		covered_C = (asg.in_R[:,None] & sure_RC).any(0)
		if covered_C.all():
			return None

		nW_RC = (self.graph_RWC & avail_W[None,:,None]).sum(1)
		per_RC = np.where(sure_RC, 1, nW_RC) * cand_R[:,None]
		nchild_C = np.where(covered_C, np.iinfo(np.int64).max, per_RC.sum(0))

		best = np.flatnonzero(nchild_C == nchild_C.min())
		return best[self.rng.integers(len(best))]

	def all_determined(self, asg: Assignment) -> bool:
		return np.all(asg.R != UND) and np.all(asg.W != UND)

	def at_target(self, asg: Assignment) -> bool:
		return (
			asg.extraR == self.num_extraR and 
			asg.extraW == self.num_extraW
		)

	def order(self, children):
		return [children[i] for i in self.rng.permutation(len(children))]

	def search(self, asg: Assignment) -> Result:
		self.search_iters += 1
		asg = self.propagate(asg)

		if asg is None:
			return Result(Status.INFEASIBLE, asg)

		# need this because there is num_extraR and num_extraW 
		if asg.extraR > self.num_extraR or asg.extraW > self.num_extraW:
			return Result(Status.INFEASIBLE, asg)

		c = self.pick_uncovered_column(asg)
		if c is None:
			if (asg.extraR == self.num_extraR and asg.extraW == self.num_extraW):
				return Result(Status.SOLVED, asg)
			if np.all(asg.R != UND) and np.all(asg.W != UND):
				return Result(Status.INFEASIBLE, asg)
			isRchild, var_idx = self.var_index(asg)
			children = self.var_children(asg, isRchild, var_idx)
		else:
			children = self.column_children(asg, c)

		for child in self.order(children):
			res = self.search(child)
			# print(res)
			if res.status == Status.SOLVED:
				return res

		return Result(Status.INFEASIBLE, asg)

	def update_stack(self, cfg: Config, stack: list[Frame]) -> list[Frame]:
		if cfg.resume_backtrack:
			return self.fixed_backtrack(cfg, stack)
		return self.populate_node(cfg, stack)

	def populate_node(self, cfg: Config, stack: list[Frame]) -> list[Frame]:
		fr = stack[-1]
		assert fr.perm is None, "node already initialized"
		# this is the first time this search node is being entered
		fr.asg = self.propagate(fr.asg)

		if fr.asg is None:  # ~unit_found
			return self.fixed_backtrack(cfg, stack)

		if fr.asg.extraR > self.num_extraR or fr.asg.extraW > self.num_extraW:  # over_target
			return self.fixed_backtrack(cfg, stack)

		# perm and one of column or var_idx is set here
		fr.child_idx = 0
		fr.column = self.pick_uncovered_column(fr.asg)
		if fr.column is None:
			if fr.asg.extraR == self.num_extraR and fr.asg.extraW == self.num_extraW: # at_target
				cfg.status = Status.SOLVED
				return stack
			needR = self.num_extraR - fr.asg.extraR
			needW = self.num_extraW - fr.asg.extraW
			if fr.asg.und_R.sum() < needR or fr.asg.und_W.sum() < needW: # over_budget
				return self.fixed_backtrack(cfg, stack)
			fr.isRchild, fr.var_idx = self.var_index(fr.asg)
			children = self.var_children(fr.asg, fr.isRchild, fr.var_idx)
			fr.perm = np.arange(len(children)) 
		else:
			children = self.column_children(fr.asg, fr.column)
			fr.perm = self.rng.permutation(len(children))

		return self.visit_child(stack)
	
	def visit_child(self, stack: list[Frame]) -> list[Frame]:
		fr = stack[-1]
		p = fr.perm[fr.child_idx]
		if fr.column is not None:
			ch = self.column_child(fr.asg, fr.column, p)
		else:
			ch = self.var_child(fr.asg, fr.isRchild, fr.var_idx, p)
		stack.append(Frame(ch))
		return stack

	def fixed_backtrack(self, cfg: Config, stack: list[Frame]) -> list[Frame]:
		"""
		Backtrack at most n times, checking for a next unvisited child at each node.
		If one exists, update the parent's child_idx to point to that child.
		Set the parent's `resume_backtrack` flag to indicate whether more backtracking is
		pending.
		"""
		step = 0
		while stack and step < self.max_backtrack_steps:
			fr = stack[-1]
			if (step == 0 and not cfg.resume_backtrack):
				stack.pop()
				step += 1
			elif fr.child_idx >= fr.perm.size - 1:
				stack.pop()
				step += 1
			else:
				fr.child_idx += 1
				cfg.resume_backtrack = False # finished
				return self.visit_child(stack)

		if stack:
			cfg.resume_backtrack = True
		return stack

	def recursive_search(self) -> tuple[Result, int]:
		self.search_iters = 0
		R, W, _ = self.graph_RWC.shape
		asg = Assignment(R, W)
		result = self.search(asg)
		return result, self.search_iters

	def loop_search(self) -> tuple[Result, int]:
		R, W, _ = self.graph_RWC.shape
		asg = Assignment(R, W)
		stack = [Frame(asg)]
		cfg = Config(status=Status.RUNNING, resume_backtrack=False)
		step = 0
		while step < self.max_search_iters and stack and cfg.status == Status.RUNNING:
			fr = stack[-1]
			stack = self.update_stack(cfg, stack)
			# print(f"{step=}, {fr.infeasible=}, {len(stack)=}, {fr.child_idx=}")
			step += 1

		if stack:
			fr = stack[-1]
			return cfg, Result(cfg.status, fr.asg), len(stack), step
		return cfg, Result(Status.INFEASIBLE, None), len(stack), step 
		

	def validate(self, asg: Assignment) -> bool:
		marg_RC = self.graph_RWC[*np.ix_(asg.in_R, asg.in_W),:].any(1)
		# print("marg_RC:\n", marg_RC.astype(np.int32))
		return bool((marg_RC.sum(0) == 1).all())

