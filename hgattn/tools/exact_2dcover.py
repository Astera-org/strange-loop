import fire
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int, Int8, Int32, Array, Bool, PRNGKeyArray, Key
from typing import NamedTuple, Self
from functools import partial
from .graph import gen_witness_graph, gen_factored_witness_graph

RUNNING = 0 
FOUND = 1 
INFEASIBLE = 2 

IN = 0 
OUT = 1 
UND = 2 

class Assignment(NamedTuple):
	state_S: Int8[Array, "S"]     # R ++ W 
	woff: int                     # offset into state_S where W vector starts

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

class Node(NamedTuple):
	asg: Assignment
	col: Int
	var: Int
	child_idx: Int
	n_child: Int
	key: PRNGKeyArray

class State(NamedTuple):
	status: Int8[Array, ""]
	state_S: Int8[Array, "S"]     # R + W state in {UND, IN, OUT}
	trail_inds_S: Int[Array, "S"] # inds into state_S that have been changed
	trail_offs_D: Int[Array, "D"] # marks the end of populated range in trail_inds_S
	extra_D2: Int[Array, "D 2"]   # extra_D2[d] = [Rextra, Wextra] 
	col_D: Int[Array, "D"]        # column, or -1 if var Node 
	var_D: Int[Array, "D"]        # UND variable index in [0, R+W) or -1 if column Node
	n_child_D: Int[Array, "D"]    # number of children (-1 if not yet visited)
	child_idx_D: Int[Array, "D"]  # next child to try, in [0, n_child_D[d])
	key_D: Key[Array, "D"]
	top: Int                      # the top of the stack

	@classmethod
	def empty(cls, D: int, R: int, W: int, key: PRNGKeyArray):
		return cls(
				status=jnp.int8(RUNNING),
				state_S=jnp.full((R + W,), UND, jnp.int8),
				trail_inds_S=jnp.full((R + W,), -1, jnp.int32),
				trail_offs_D=jnp.full((R + W,), 0, jnp.int32),
				extra_D2=jnp.full((D, 2), 0, jnp.int32),
				col_D=jnp.full((D,), -1, jnp.int32),
				var_D=jnp.full((D,), -1, jnp.int32),
				n_child_D=jnp.full((D,), -1, jnp.int32),
				child_idx_D=jnp.zeros((D,), jnp.int32),
				key_D=jnp.zeros((D,), key.dtype),
				top=jnp.int32(-1)
				)

	def push_state(self, state_S) -> Self:
		"""pushes the state, keeping an undo trail"""
		S = state_S.shape[0]
		mask = (state_S != self.state_S)
		off = self.trail_offs_D[self.top]
		rank = jnp.cumsum(mask) - 1
		pos = jnp.where(mask, off + rank, S)

		return self._replace(
				state_S=state_S,
				trail_inds_S=self.trail_inds_S.at[pos].set(jnp.arange(S), mode="drop"),
				trail_offs_D=self.trail_offs_D.at[self.top+1].set(off + mask.sum()),
				top=self.top + 1,
				)

	def pop_state(self) -> Self:
		"""restore previous state using the undo trail"""
		S = self.state_S.shape[0]
		rng = jnp.arange(S)
		beg = self.trail_offs_D[self.top-1]
		end = self.trail_offs_D[self.top]
		pos = jnp.where((beg <= rng) & (rng < end), self.trail_inds_S, S)

		return self._replace(
				state_S=self.state_S.at[pos].set(UND, mode="drop"),
				top=self.top-1,
				)


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
			):
		# if validW_RKC[r,k,c], then w = indsW_RKC[r,k,c] defines (r,w,c) as edge in the graph
		self.indsW_RKC, self.validW_RKC = compress_graph(graph_RWC) 
		self.num_extraR = extraR
		self.num_extraW = extraW
		self.R, self.W, self.C = graph_RWC.shape
		self.K = self.indsW_RKC.shape[1]
		self.D = self.R + self.W # maybe too much

	def push(self, st, node, j) -> State:
		"""push node as the j'th child of the node at the top of the stack"""
		top = st.top + 1
		return st._replace(
				stack_DR=st.stack_DR.at[top,:].set(node.asg.R),
				stack_DW=st.stack_DW.at[top,:].set(node.asg.W),
				col_D=st.col_D.at[top].set(node.col),
				var_D=st.var_D.at[top].set(node.var),
				child_idx_D=st.child_idx_D.at[top].set(j),
				key_D=st.key_D.at[top].set(node.key),
				top=top,
				)

	def propagate(asg: Assignment) -> tuple[Assignment, Bool]:

		def body_fn(carry):
			asg, more, _ = carry

			sure_RC = jnp.where(self.validW_RKC, asg.in_W[self.indsW_RKC], 0).any(1)
			poss_RC = jnp.where(self.validW_RKC, ~asg.out_W[self.indsW_RKC], 0).any(1)

			cover_C = (asg.in_R[:,None] & sure_RC).sum(0)
			cand_C = (~asg.out_R[:,None] & poss_RC).sum(0)

			ok = ~((cover_C >= 2).any() | (cand_C == 0).any())
			covered_C = (cover_C == 1)

			out_R = asg.und_R & (sure_RC & covered_C[None,:]).any(1)

			# witnesses whose addition would double-cover some column
			in_sure_RC = asg.in_R[:,None] & sure_RC
			wts = self.validW_RKC * (1 - in_sure_RC)[:,None,:]
			inds_C_RKC = jnp.broadcast_to(jnp.arange(C), (R, K, C))
			out_WC = jnp.zeros((W, C), dtype=jnp.int32).at[self.indsW_RKC,indsC_RKC].add(wts)
			newcov_WC = out_WC + in_sure_RC.sum(0)[None,:]
			out_W = asg.und_W & (newcov_WC >= 2).any(1)

			in_R, in_W = self.unit_rules(asg, poss_RC, cand_C, covered_C)

			ok = ok & ~((in_R & out_R).any() | (in_W & out_W).any())

			more = (out_R | in_R).any() | (out_W | in_W).any()

			in_S = jnp.concat((in_R, in_W))
			out_S = jnp.concat((out_R, out_W))
			asg = asg._replace(state_S=asg.state_S[in_S].set(IN).at[out_S].set(OUT))

			return asg, more

		def cond_fn(carry):
			_, more, ok = carry
			return more & ok

		init_carry = state, jnp.bool(True), jnp.bool(True)
		state, _, ok = jax.lax.while_loop(cond_fn, body_fn, init_carry)
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
		indsC_RKC = jnp.broadcast_to(jnp.arange(C), (R, K, C))
		hit_WC = jnp.zeros((W, C), dtype=jnp.int32).at[self.indsW_RKC, indsC_RKC].max(vals_RKC)
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

		viableW_RK = validW_RK & (asg.R!= OUT)[:,None] & (W_RK != OUT) 
		singles_R = (viableW_RK & (W_RK == IN)).any(1)
		counts_R = jnp.where(singles_R, 1, viable_RK.sum(1))

		return indsW_RK, validW_RK, singles_R, counts_R

	def num_column_children(self, asg, c):
		"""Compute the number of viable column children"""
		return self._column_options(asg, c)[3].sum()

	def _column_node_child(self, node) -> Assignment:
		"""the `j`th child (0 <= j < n_child) in order fixed by the frame's key"""
		indsW_RK, validW_RK, singles_R, counts_R = self._column_options(node.asg, c)
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
		prio_K = jnp.where(viable_RK[r],
					 jax.random.uniform(jax.random.fold_in(kW, r), (self.K,)),
					 jnp.inf)
		rank_K = jnp.argsort(jnp.argsort(prio_K))
		pick_K = viable_RK[r] & (rank_K == k)
		out_K = viable_RK[r] & (rank_K < k)

		inds = indsW_RK[r]
		W_new = node.asg.W.at[jnp.where(pick_K, inds, self.W)].set(IN, mode="drop")
		W_new = W_new.at[jnp.where(out_K, inds, self.W)].set(OUT, mode="drop")
		W_new = jnp.where(singles_R[r], node.asg.W, W_new)  # row-only child has no witness change

		return Assignment(jnp.concat((R_new, W_new)), self.R) 

	def _pick_var(self, asg: Assignment, key) -> Int32:
		und = jnp.concat((asg.und_R, asg.und_W))
		logits = jnp.where(und, 0.0, -jnp.inf)
		return jax.random.categorical(key, logits)

	def _var_node_child(self, node) -> Assignment:
		"""Choose the child of this node based on the child_idx"""
		perm = jax.random.permutation(node.key, 2)
		val = jnp.where(perm[node.child_idx], jnp.int8(IN), jnp.int8(OUT))
		is_r = (node.var < self.R)
		idx = jnp.where(is_r, node.var, node.var - R)
		R_new = jnp.where(is_r, node.R.at[idx].set(val), node.R)
		W_new = jnp.where(is_r, node.W, node.W.at[idx].set(val))

		return Assignment(jnp.concat(R_new, W_new), self.R)

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
		nchild_C = jnp.where(covered_C, jnp.iinfo(jnp.int64).max, per_RC.sum(0))

		best = jnp.argmin(nchild_C)
		return best, ~covered_C.all()

	def initial_state(self, key: PRNGKeyArray) -> State:
		pass

	def update_stack(self, state: State) -> State:
		in_backtrack = state.in_backtrack_D[state.top]
		prev_visited = state.n_child_D[state.top] >= 0

		asg = Assignment(st.state_S, self.R)
		asg, unit_found = self.propagate(asg)

		col, col_found = self.pick_uncovered_column(asg)
		var = jnp.where(col_found, -1, self.pick_var(asg, key))

		n_child = jax.lax.cond(col_found, self.num_column_children(asg, col), 2)
		child_idx = jnp.where(prev_visited, state.child_idx_D[state.top], 0)


		state = jax.lax.cond(bad, self.fixed_backtrack, lambda x: x, state)

		bad = in_backtrack | ~unit_found
		solved = ~col_found & at_target 

		at_target = asg.extraR == self.num_extraR & asg.extraW == self.num_extraW
		over_target = asg.extraR > self.num_extraR | asg.extraW > self.num_extraW
		all_chosen = (asg.R != UND).all() & (asg.W != UND).all()
		last_child = child_idx >= n_child

		node = Node(asg, col, var, child_idx, n_child)
		child_asg = jax.lax.cond(col_found, self._column_node_child, self._var_node_child, node)

		def _visit_child(st):
			return st.push_state(child_asg.state_S)

		infeasible = (
				in_backtrack 
				| ~unit_found
				| over_target 
				| (~col_found & all_chosen & ~at_target)
				| last_child
				)

		return jax.lax.cond(infeasible, self.fixed_backtrack, _visit_child, state)


	def fixed_backtrack(self, state: State) -> State:
		"""
		Backtrack at most n times, checking for a next unvisited child at each node.
		If one exists, update the parent's loop_idx to point to that child.
		Set the parent's `in_backtrack` flag to indicate whether more backtracking is
		pending.

		Use the n_child_D[d] >= 0 to tell whether the node has been entered
		"""
		def advance(st):
			return st._replace(
					child_idx_D=st.child_idx_D.at[st.top].add(1),
					in_backtrack_D=st.in_backtrack_D.at[st.top].set(False),
					)

		def body_fn(i, st):
			entered = (st.n_child_D[st.top] >= 0)
			last_child = (st.child_idx_D[st.top] >= st.n_child_D[st.top] - 1)
			st = jax.lax.cond(~entered | last_child, State.pop_state, advance, st)
			return st

		state = jax.lax.fori_loop(0, self.max_backtrack_steps, body_fn, state, unroll=True)

		def set_backtrack(st):
			return st._replace(in_backtrack_D=st.in_backtrack_D.at[st.top].set(True))
		return jax.lax.where(state.top == -1, state, set_backtrack(state)) 


	def _loop_search(self, key: PRNGKeyArray) -> State:
		def cond_fn(st):
			return st.status == RUNNING

		init_state = State.empty(self.D, self.R, self.W, key)
		return jax.lax.while_loop(cond_fn, self.update_stack, init_state)

	def search(self, key: PRNGKeyArray) -> Result:
		state = self._loop_search(key)
		valid = state.top >= 0 & state.status == FOUND
		asg = Assignment(state.state_S, self.R)
		return Result(asg, valid)

	def validate(self, asg: Assignment) -> Bool[Array, ""]:
		sure_RKC = jnp.where(
				asg.in_R[:,None,None],
				jnp.where(self.validW_RKC, asg.in_W[self.indsW_RKC], 0),
				0
				)
		marg_RC = sure_RKC.any(1)
		return jnp.all(marg_RC.sum(0) == 1)


def main(
		R: int, 
		W: int, 
		C: int,
		w_per_row: int,
		w_per_col: int,
		w_per_link: int,
		seed: int,
		num_extra_rows: int,
		num_extra_witnesses: int,
		use_factored: bool,
		):
	jnp.set_printoptions(linewidth=180, threshold=10000000)
	key = jax.random.key(seed)
	rng = np.random.default_rng(seed)

	if use_factored:
		graph_RWC = gen_factored_witness_graph(R, W, C, w_per_row, w_per_col, rng)
	else:
		graph_RWC = gen_witness_graph(R, W, C, w_per_link, rng)
	graph_RWC = jnp.array(graph_RWC)
	cover2D = ExactCover2D(graph_RWC, num_extra_rows, num_extra_witnesses)

	print(f"Graph Density: {(graph_RWC.sum() / graph_RWC.size):5.3f}")

	total = 10
	num_valid = 0

	for i in range(total):
		sub_key = jax.random.fold_in(key, i)
		res = cover2D.search(sub_key)
		is_valid = cover2D.validate(res.asg)
		print(f"{i}: {is_valid=}, {res.asg}")



if __name__ == "__main__":
	fire.Fire(main)


