import fire
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses
from jaxtyping import Int, Int8, Int32, Array, Bool, PRNGKeyArray, Key
import equinox as eqx
from typing import NamedTuple, Self
from functools import partial
from .graph import gen_witness_graph, gen_factored_witness_graph

FOUND = 0 
RUNNING = 1 
INFEASIBLE = 2

IN = 0 
OUT = 1 
UND = 2 

class Assignment(eqx.Module):
	woff: int = eqx.field(static=True)  # offset into state_S where W vector starts
	state_S: Int8[Array, "S"]           # R ++ W 
	extraR: Int[Array, ""]
	extraW: Int[Array, ""]

	def __init__(self, woff: int, state: Array, extraR: Array, extraW: Array):
		self.woff = woff
		self.state_S = state
		self.extraR = extraR 
		self.extraW = extraW 

	@property
	def R(self):
		return self.state_S[:self.woff]

	@property
	def W(self):
		return self.state_S[self.woff:]

	@property
	def in_R(self):
		return self.R == IN

	@property
	def out_R(self):
		return self.R == OUT

	@property
	def und_R(self):
		return self.R == UND

	@property
	def in_W(self):
		return self.W == IN

	@property
	def out_W(self):
		return self.W == OUT

	@property
	def und_W(self):
		return self.W == UND

class Result(NamedTuple):
	asg: Assignment
	valid: Bool[Array, ""] 
	step: int
	depth: int

class Node(NamedTuple):
	asg: Assignment
	col: Int
	var: Int
	child_idx: Int
	n_child: Int
	key: PRNGKeyArray

class State(eqx.Module):
	status: Int8[Array, ""]
	resume_backtrack: Bool[Array, ""]
	assignment_S: Int8[Array, "S"]     # R + W state in {UND, IN, OUT}
	trail_inds_S: Int[Array, "S"] # inds into assignment_S that have been changed
	trail_offs_D: Int[Array, "D"] # marks the end of populated range in trail_inds_S
	extra_D2: Int[Array, "D 2"]   # extra_D2[d] = [Rextra, Wextra] 
	col_D: Int[Array, "D"]        # column, or -1 if var Node 
	var_D: Int[Array, "D"]        # UND variable index in [0, R+W) or -1 if column Node
	n_child_D: Int[Array, "D"]    # number of children (-1 if not yet visited)
	child_idx_D: Int[Array, "D"]  # next child to try, in [0, n_child_D[d])
	key_D: Key[Array, "D"]
	top: Int[Array, ""]           # the top of the stack
	R: int = eqx.field(static=True)

	@classmethod
	def root(cls, D: int, R: int, W: int, key: PRNGKeyArray):
		"""
		Return the root node state
		"""
		return cls(
				status=jnp.int8(RUNNING),
				resume_backtrack=jnp.bool(False),
				assignment_S=jnp.full((R + W,), UND, jnp.int8),
				trail_inds_S=jnp.full((R + W,), -1, jnp.int32),
				trail_offs_D=jnp.full((D,), 0, jnp.int32),
				extra_D2=jnp.full((D, 2), 0, jnp.int32),
				col_D=jnp.full((D,), -1, jnp.int32),
				var_D=jnp.full((D,), -1, jnp.int32),
				n_child_D=jnp.full((D,), -1, jnp.int32),
				child_idx_D=jnp.zeros((D,), jnp.int32),
				key_D=jnp.zeros((D,), key.dtype).at[0].set(key),
				top=jnp.int32(0),
				R=R,
				)
	
	def get_assignment(self) -> Assignment:
		extraR, extraW = self.extra_D2[self.top]
		return Assignment(self.R, self.assignment_S, extraR, extraW) 

	def get_node(self) -> Node:
		asg = self.get_assignment()
		col = self.col_D[self.top]
		var = self.var_D[self.top]
		n_child = self.n_child_D[self.top]
		child_idx = self.child_idx_D[self.top]
		key = self.key_D[self.top]
		return Node(asg, col, var, child_idx, n_child, key)
	
	def update_assignment(
		self, 
		asg: Assignment, 
		in_frame: Bool[Array, ""]
	) -> Self:
		"""
		updates the assignment_S field, keeping an undo trail.
		If `in_frame`, 
		"""
		S = asg.state_S.shape[0]
		mask = (asg.state_S != self.assignment_S)
		off = self.trail_offs_D[self.top]
		rank = jnp.cumsum(mask) - 1
		pos = jnp.where(mask, off + rank, S)
		frame = jnp.where(in_frame, self.top, self.top + 1)

		extra = jnp.array([asg.extraR, asg.extraW])

		return dataclasses.replace(
			self,
			assignment_S=asg.state_S,
			trail_inds_S=self.trail_inds_S.at[pos].set(jnp.arange(S), mode="drop"),
			trail_offs_D=self.trail_offs_D.at[frame].set(off + mask.sum()),
			extra_D2=self.extra_D2.at[frame].set(extra),
			top=frame,
		)
	
	def update_frame(self, node: Node) -> Self:
		"""update the top frame with content of node"""
		return dataclasses.replace(
			self,
			col_D=self.col_D.at[self.top].set(node.col),
			var_D=self.var_D.at[self.top].set(node.var),
			child_idx_D=self.child_idx_D.at[self.top].set(node.child_idx),
			n_child_D=self.n_child_D.at[self.top].set(node.n_child),
		)

	def set_key(self, key: PRNGKeyArray) -> Self:
		new_key_D = self.key_D.at[self.top].set(key)
		return dataclasses.replace(self, key_D=new_key_D)

	def pop_state(self) -> Self:
		"""restore previous state using the undo trail.  Stack must be non-empty"""
		S = self.assignment_S.shape[0]
		rng = jnp.arange(S)
		beg = jnp.where(self.top == 0, 0, self.trail_offs_D[self.top-1])
		end = self.trail_offs_D[self.top]
		pos = jnp.where((beg <= rng) & (rng < end), self.trail_inds_S, S)

		def _do_pop(st):
			return dataclasses.replace(
				st,
				assignment_S=st.assignment_S.at[pos].set(UND, mode="drop"),
				n_child_D=st.n_child_D.at[st.top].set(-1),
				child_idx_D=st.child_idx_D.at[st.top].set(0),
				top=st.top-1,
			)
		return _do_pop(self)


def compress_graph(
		graph_RWC: Int8[Array, "R W C"],
		) -> tuple[Int[Array, "R K C"], Bool[Array, "R K C"]]:
	"""
	Return a W-index representation of the graph.
	Cannot be jitted due to non-static size
	"""
	max_occu = graph_RWC.sum(1).max()
	inds_RWC = jnp.argsort(-graph_RWC, axis=1)
	vals_RWC = jnp.sort(-graph_RWC, axis=1)
	masked_RWC = jnp.where(vals_RWC == -1, inds_RWC, -1)
	inds_RWC = jnp.where(vals_RWC == -1, inds_RWC, 0)
	valid_RWC = (vals_RWC == -1)
	return inds_RWC[:,:max_occu,:], valid_RWC[:,:max_occu,:]

class ExactCover2D:

	def __init__(
			self,
			graph_RWC: Int8[Array, "R W C"],
			extraR: int,
			extraW: int,
			max_backtrack_steps: int,
			max_propagate_steps: int,
			max_search_steps: int
			):
		# if validW_RKC[r,k,c], then w = indsW_RKC[r,k,c] defines (r,w,c) as edge in the graph
		self.indsW_RKC, self.validW_RKC = compress_graph(graph_RWC) 
		self.num_extraR = extraR
		self.num_extraW = extraW
		self.R, self.W, self.C = graph_RWC.shape
		self.K = self.indsW_RKC.shape[1]
		self.D = self.R + self.W + 1 
		self.max_propagate_steps = max_propagate_steps 
		self.max_backtrack_steps = max_backtrack_steps 
		self.max_search_steps = max_search_steps

	def propagate(self, asg: Assignment) -> tuple[Assignment, Bool]:

		def body_fn(i, carry):
			asg, prev_ok = carry

			# (R,C) combos that are surely supported by a witness 
			sure_RC = jnp.where(self.validW_RKC, asg.in_W[self.indsW_RKC], 0).any(1)

			# (R,C) combos that may eventually be supported by a witness (or more) 
			poss_RC = jnp.where(self.validW_RKC, ~asg.out_W[self.indsW_RKC], 0).any(1)

			cover_count_C = (asg.in_R[:,None] & sure_RC).sum(0)
			cand_C = (~asg.out_R[:,None] & poss_RC).sum(0)

			ok = ~((cover_count_C >= 2).any() | (cand_C == 0).any())
			covered_C = (cover_count_C == 1)

			# forced-OUT rows, previously UND but clashing with an already covered column 
			out_R = asg.und_R & (sure_RC & covered_C[None,:]).any(1)

			# newcov_WC = einsum('r,rwc->wc', st.IN_R, sure_RC[:,None,:] | g_RWC)

			indsC_RKC = jnp.broadcast_to(jnp.arange(self.C), (self.R, self.K, self.C))
			zeros_WC = jnp.zeros((self.W, self.C), dtype=jnp.int32)

			extra_WC = zeros_WC.at[self.indsW_RKC, indsC_RKC].add(
				asg.in_R[:,None,None] & self.validW_RKC & ~sure_RC[:,None,:]
			)

			# count of rows supporting (W,C) interaction 
			newcov_WC = cover_count_C[None,:] + extra_WC

			# forced-OUT witnesses, previously UND but inducing multiple rows
			# supporting the (W,C) interaction 
			out_W = asg.und_W & (newcov_WC >= 2).any(1)
			out_S = jnp.concat((out_R, out_W))

			new_state_S = jnp.where(out_S, OUT, asg.state_S)
			asg = eqx.tree_at(lambda m: m.state_S, asg, new_state_S)
			# jax.debug.breakpoint()

			in_R, in_W = self.unit_rules(asg, poss_RC, cand_C, covered_C)

			ok = ok & ~((in_R & out_R).any() | (in_W & out_W).any())
			# jax.debug.breakpoint()

			in_S = jnp.concat((in_R, in_W))
			out_S = jnp.concat((out_R, out_W))
			new_state_S = jnp.where(out_S, OUT, jnp.where(in_S, IN, asg.state_S))
			asg = eqx.tree_at(lambda m: m.state_S, asg, new_state_S) 

			return asg, ok & prev_ok

		init_carry = asg, jnp.bool(True)
		state, ok = jax.lax.fori_loop(0, self.max_propagate_steps, body_fn, init_carry)
		return state, ok

	def unit_rules(
			self,
			asg: Assignment,
			poss_RC: Array,
			cand_C: Array,
			covered_C: Array,
			) -> tuple[Array, Array]:
		"""
		Unit-propagation for forced-in R and W
		"""
		open_C = ~covered_C

		# Candidate rows for each uncovered column: not OUT, with a possible edge.
		cand_RC = ~asg.out_R[:,None] & poss_RC & open_C[None,:]

		# Row unit: an uncovered column with exactly one candidate row forces that row IN
		unit_C = open_C & (cand_C == 1)
		in_R = asg.und_R & (cand_RC & unit_C[None,:]).any(1)

		# Witness unit: if every candidate path into an uncovered column c
		# goes through the same single non-OUT witness, that witness is forced IN
		vals_RKC = (self.validW_RKC & cand_RC[:,None,:]).astype(jnp.int32)
		indsC_RKC = jnp.broadcast_to(jnp.arange(self.C), (self.R, self.K, self.C))
		zeros_WC = jnp.zeros((self.W, self.C), dtype=jnp.int32)
		hit_WC = zeros_WC.at[self.indsW_RKC, indsC_RKC].max(vals_RKC)
		path_WC = (hit_WC > 0) & ~asg.out_W[:,None]
		# path_WC = (cand_RC[:,None,:] & self.graph_RWC & ~asg.out_W[None,:,None]).any(0)
		nW_C = path_WC.sum(0)
		unitW_C = open_C & (nW_C == 1)
		in_W = asg.und_W & (path_WC & unitW_C[None,:]).any(1)

		return in_R, in_W

	def _column_options(self, asg, c):
		"""
		Recomputed per child.
		"""
		indsW_RK = self.indsW_RKC[:,:,c]
		validW_RK = self.validW_RKC[:,:,c]

		W_RK = asg.W[indsW_RK] # padded slots; masked below

		viableW_RK = validW_RK & (asg.R != OUT)[:,None] & (W_RK != OUT) 
		singles_R = (viableW_RK & (W_RK == IN)).any(1)
		counts_R = jnp.where(singles_R, 1, viableW_RK.sum(1))

		return indsW_RK, viableW_RK, singles_R, counts_R

	def num_column_children(self, asg, c):
		"""Compute the number of viable column children"""
		return self._column_options(asg, c)[3].sum()

	def _column_child(self, node) -> Assignment:
		"""the `j`th child (0 <= j < n_child) in order fixed by the frame's key"""
		indsW_RK, viableW_RK, singles_R, counts_R = self._column_options(node.asg, node.col)
		kR, kW = jax.random.split(node.key)

		perm_R = jax.random.permutation(kR, self.R)
		cnt = counts_R[perm_R] # some of these will be zero
		end = jnp.cumsum(cnt)
		i = jnp.searchsorted(end, node.child_idx, side="right")
		k = node.child_idx - (end[i] - cnt[i]) # index within row r
		r = perm_R[i]

		R_new = node.asg.R.at[r].set(IN)

		# Witness order on W(r,c): a function of (key, r) only, so it is the same
		# for every child of r in this frame.
		prio_K = jnp.where(
			viableW_RK[r],
			jax.random.uniform(jax.random.fold_in(kW, r), (self.K,)),
			jnp.inf)
		rank_K = jnp.argsort(jnp.argsort(prio_K))
		pick_K = viableW_RK[r] & (rank_K == k)
		out_K = viableW_RK[r] & (rank_K < k)

		inds = indsW_RK[r]
		W_new = node.asg.W.at[jnp.where(pick_K, inds, self.W)].set(IN, mode="drop")
		W_new = W_new.at[jnp.where(out_K, inds, self.W)].set(OUT, mode="drop")
		W_new = jnp.where(singles_R[r], node.asg.W, W_new)  # row-only child has no witness change
		new_state_S = jnp.concat((R_new, W_new))
		return eqx.tree_at(lambda m: m.state_S, node.asg, new_state_S)

	def _pick_var(self, asg: Assignment, key) -> Int32:
		und = jnp.concat((asg.und_R, asg.und_W))
		logits = jnp.where(und, 0.0, -jnp.inf)
		return jax.random.categorical(key, logits)

	def _var_child(self, node) -> Assignment:
		"""Choose the child of this node based on the child_idx"""
		val = jnp.where(node.child_idx == 0, jnp.int8(IN), jnp.int8(OUT))
		is_r = (node.var < self.R)
		idx = jnp.where(is_r, node.var, node.var - self.R)
		asg = node.asg
		R_new = jnp.where(is_r, asg.R.at[idx].set(val), asg.R)
		W_new = jnp.where(is_r, asg.W, asg.W.at[idx].set(val))
		S_new = jnp.concat((R_new, W_new))
		extraR_new = asg.extraR + (is_r & (val == IN))
		extraW_new = asg.extraW + (~is_r & (val == IN))

		return eqx.tree_at(
			lambda m: (m.state_S, m.extraR, m.extraW),
			node.asg,
			(S_new, extraR_new, extraW_new)
		)

	def pick_uncovered_column(self, asg) -> tuple[Array, Array]:
		"""
		Returns:
		  uncovered column with fewest column_children
		  found: boolean Array whether column was found
		"""
		cand_R = (asg.R != OUT)
		avail_W = (asg.W != OUT)

		sure_RC = jnp.where(self.validW_RKC, asg.in_W[self.indsW_RKC], 0).any(1)
		covered_C = (asg.in_R[:,None] & sure_RC).any(0)

		nW_RC = jnp.where(self.validW_RKC, avail_W[self.indsW_RKC], 0).any(1)
		per_RC = jnp.where(sure_RC, 1, nW_RC) * cand_R[:,None]
		nchild_C = jnp.where(covered_C, jnp.iinfo(jnp.int32).max, per_RC.sum(0))

		best = jnp.argmin(nchild_C)
		return best, ~covered_C.all()

	def initial_state(self, key: PRNGKeyArray) -> State:
		pass

	def update_stack(self, step: Int[Array, ""], state: State) -> State:
		asg = state.get_assignment()
		"""
		jax.debug.print(
			"top: {}, "
			"child_idx: {}, "
			"status: {}, "
			"asg.extraR: {}, "
			"asg.extraW: {}, ",
			state.top,
			state.child_idx_D[state.top],
			state.status,
			asg.extraR,
			asg.extraW,
			ordered=True
		)
		"""

		return jax.lax.cond(
			state.resume_backtrack, self.fixed_backtrack, self.populate_node, state)

	def populate_node(self, state: State) -> State:
		asg = state.get_assignment()
		asg, unit_found = self.propagate(asg)
		col, col_found = self.pick_uncovered_column(asg)
		key = state.key_D[state.top]
		node_var = jnp.where(col_found, -1, self._pick_var(asg, key))
		node_col = jnp.where(col_found, col, -1)

		child_idx = 0 
		n_child = jnp.where(col_found, self.num_column_children(asg, col), 2)
		node = Node(asg, node_col, node_var, child_idx, n_child, key)
		state = state.update_frame(node)
		state = state.update_assignment(asg, in_frame=True)

		# assignment related flags
		over_target = (asg.extraR > self.num_extraR) | (asg.extraW > self.num_extraW)
		at_target = (asg.extraR == self.num_extraR) & (asg.extraW == self.num_extraW)
		found = ~col_found & unit_found & at_target 

		needR = self.num_extraR - asg.extraR
		needW = self.num_extraW - asg.extraW
		over_budget = (asg.und_R.sum() < needR) | (asg.und_W.sum() < needW)

		def _set_found(st):
			return dataclasses.replace(st, status=jnp.int8(FOUND))

		infeasible = ~unit_found | over_target | (~col_found & over_budget)

		# jax.debug.print("2: switch_idx: {}", switch_idx, ordered=True)

		state = jax.lax.switch(
			jnp.argmax(jnp.array([found, infeasible, True])),
			[_set_found, self.fixed_backtrack, self.visit_child],
			state
		)

		return state

	def visit_child(self, state: State) -> State:
		node = state.get_node()
		child_asg = jax.lax.cond(node.col == -1, self._var_child, self._column_child, node)
		child_key = jax.random.fold_in(node.key, node.child_idx)
		state = state.update_assignment(child_asg, in_frame=False)
		state = state.set_key(child_key)
		return state

	def fixed_backtrack(self, state: State) -> State:
		"""
		Backtrack at most n times, checking for a next unvisited child at each node.
		If one exists, update the parent's loop_idx to point to that child.
		Set the parent's `resume_backtrack` flag to indicate whether more backtracking is
		pending.
		"""
		n = self.max_backtrack_steps

		def body_fn(step, carry):
			st, found = carry
			empty = st.top < 0
			t = jnp.maximum(st.top, 0)
			last_child = (st.child_idx_D[t] >= st.n_child_D[t] - 1)
			forced = (step == 0) & ~st.resume_backtrack
			halt = found | empty 
			do_pop = ~halt & (forced | last_child)
			do_find = ~halt & ~do_pop
			st = jax.lax.cond(do_pop, State.pop_state, lambda s: s, st)
			return st, found | do_find

		st, found  = jax.lax.fori_loop(0, n, body_fn, (state, jnp.bool(False)))
		
		def advance_fn(st):
			st = dataclasses.replace(st, child_idx_D=st.child_idx_D.at[st.top].add(1))
			return self.visit_child(st)

		st = jax.lax.cond(found, advance_fn, lambda s: s, st)
		return dataclasses.replace(st, resume_backtrack=~found & (st.top >= 0))


	def _loop_search(self, key: PRNGKeyArray) -> State:
		def cond_fn(carry):
			step, state = carry
			return (state.top >= 0) & (state.status == RUNNING) & (step < self.max_search_steps)

		def body_fn(carry):
			step, state = carry
			return step + 1, self.update_stack(step, state)

		init_state = jnp.int32(0), State.root(self.D, self.R, self.W, key)
		# return jax.lax.fori_loop(0, 20, body_fn, init_state)
		return jax.lax.while_loop(cond_fn, body_fn, init_state)

	def search(self, key: PRNGKeyArray) -> Result:
		step, state = self._loop_search(key)
		valid = (state.top >= 0) & (state.status == FOUND)
		asg = Assignment(self.R, state.assignment_S, jnp.int32(0), jnp.int32(0))
		return Result(asg, valid, step, state.top)

	def validate(self, asg: Assignment) -> Bool[Array, ""]:
		sure_RKC = jnp.where(
				asg.in_R[:,None,None],
				jnp.where(self.validW_RKC, asg.in_W[self.indsW_RKC], 0),
				0
				)
		marg_RC = sure_RKC.any(1)
		# jax.debug.breakpoint()
		# jax.debug.print("marg_RC:\n{}", marg_RC)
		return jnp.all(marg_RC.sum(0) == 1)
