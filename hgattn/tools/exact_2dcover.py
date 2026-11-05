import jax
import jax.numpy as jnp
from jaxtyping import Int, Int8, Array, PRNGKeyArray, Key
from typing import NamedTuple
from functools import partial

RUNNING = 0 
FOUND = 1 
INFEASIBLE = 2 

IN = 0 
OUT = 1 
UND = 2 

class State(NamedTuple):
    status: Int8[Array, ""]
    stack_DR: Int8[Array, "D R"]
    stack_DW: Int8[Array, "D W"]
    col_D: Int[Array, "D"]      # column, or -1 if this is a var state
    n_child_D: Int[Array, "D"]  # number of children
    cursor_D: Int[Array, "D"]   # next child to try, in [0, n_child_D[d])
    key_D: Key[Array, "D"]
    top: Int                    # the top of the stack

    @classmethod
    def empty(cls, D: int, R: int, W: int, key: PRNGKeyArray):
        return cls(
            status=jnp.int8(RUNNING),
            stack_DR=jnp.full((D, R), UND, jnp.int8),
            stack_DW=jnp.full((D, W), UND, jnp.int8),
            col_D=jnp.full((D,), -1, jnp.int32),
            n_child_D=jnp.zeros((D,), jnp.int32),
            cursor_D=jnp.zeros((D,), jnp.int32),
            key_D=jnp.zeros((D,), key.dtype),
            top=jnp.int32(-1)
        )

	@property
	def in_R(self):
		return self.stack_DR[self.top,:] == IN

	@property
	def out_R(self):
		return self.stack_DR[self.top,:] == OUT

	@property
	def und_R(self):
        return self.stack_DR[self.top,:] == UND

	@property
	def in_W(self):
        return self.stack_DW[self.top,:] == IN

	@property
	def out_W(self):
        return self.stack_DW[self.top,:] == OUT

	@property
	def und_W(self):
        return self.stack_DW[self.top,:] == UND

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
        budgetR: int,
        budgetW: int,
        max_stack_depth: int,
    ):
        self.indsW_RKC, self.validW_RKC = compress_graph(graph_RWC) 
        self.budgetR = budgetR
        self.budgetW = budgetW
        self.R, self.W, self.C = graph_RWC.shape
        self.K = self.indsW_RKC.shape[1]
        self.D = max_stack_depth

    def propagate(state: State) -> State:

        def body_fn(carry):
            st, more = carry

            sure_RC = jnp.where(self.validW_RKC, st.in_W[self.indsW_RKC], 0).any(1)
            # sure_RC = (self.graph_RWC & st.in_W[None,:,None]).any(1)

            poss_RC = jnp.where(self.validW_RKC, ~st.out_W[self.indsW_RKC], 0).any(1)
            # poss_RC = (self.graph_RWC & ~st.out_W[None,:,None]).any(1)

            cover_C = (st.in_R[:,None] & sure_RC).sum(0)
            cand_C = (~st.out_R[:,None] & poss_RC).sum(0)

            status = jnp.where(
                (cover_C >= 2).any() | (cand_C == 0).any(),
                INFEASIBLE,
                st.status
            )
            covered_C = (cover_C == 1)

            out_R = st.und_R & (sure_RC & covered_C[None,:]).any(1)

            # witnesses whose addition would double-cover some column
            in_sure_RC = st.in_R[:,None] & sure_RC
            wts = self.validW_RKC * (1 - in_sure_RC)[:,None,:]
            inds_C_RKC = jnp.broadcast_to(jnp.arange(C), (R, K, C))
            out_WC = jnp.zeros((W, C), dtype=jnp.int32).at[self.indsW_RKC,indsC_RKC].add(wts)
            newcov_WC = out_WC + in_sure_RC.sum(0)[None,:]
            out_W = st.und_W & (newcov_WC >= 2).any(1)

            in_R, in_W = self.unit_rules(st, poss_RC, cand_C, covered_C)

            status = jnp.where(
                (in_R & out_R).any() | (in_W & out_W).any(),
                INFEASIBLE,
                self.RUNNING,
                status
            )

            more = ((out_R | in_R).any() | (out_W | in_W).any()) & (status == self.RUNNING)

            st = st._replace(
                status=status,
                R=R.at[in_R].set(IN).at[out_R].set(OUT),
                W=W.at[in_W].set(IN).at[out_W].set(OUT),
            )
            return st, more

        def cond_fn(carry):
            _, more = carry
            return more

        init_carry = state, jnp.bool(True)
        state, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return state 

    def unit_rules(
        self,
        st: State,
        poss_RC: Array,
        cand_C: Array,
        covered_C: Array,
    ) -> tuple[Array, Array]:
        """
        Unit-propagation for forced-in R and W
        """
        open_C = ~covered_C

		# Candidate rows for each uncovered column: not OUT, with a possible edge.
		cand_RC = ~st.out_R[:,None] & poss_RC & open_C[None,:]

		# Row unit: an uncovered column with exactly one candidate row forces that row IN
		unit_C = open_C & (cand_C == 1)
		in_R = st.und_R & (cand_RC & unit_C[None,:]).any(1)

		# Witness unit: if every candidate path into an uncovered column c
		# goes through the same single non-OUT witness, that witness is forced IN
        vals_RKC = (self.validW_RKC & cand_RC[:,None,:]).astype(jnp.int32)
        indsC_RKC = jnp.broadcast_to(jnp.arange(C), (R, K, C))
        hit_WC = jnp.zeros((W, C), dtype=jnp.int32).at[self.indsW_RKC, indsC_RKC].max(vals_RKC)
        path_WC = (hit_WC > 0) & ~st.out_W[:,None]
		# path_WC = (cand_RC[:,None,:] & self.graph_RWC & ~st.out_W[None,:,None]).any(0)
		nW_C = path_WC.sum(0)
		unitW_C = open_C & (nW_C == 1)
		in_W = st.und_W & (path_WC & unitW_C[None,:]).any(1)

		return in_R, in_W

    def _column_options(self, st, c):
        """
        Recomputed per child.
        """
        indsW_RK = self.indsW_RKC[:,:,c]
        validW_RK = self.validW_RKC[:,:,c]

        W_RK = st.W[indsW_RK] # padded slots; masked below

        viableW_RK = validW_RK & (st.R != OUT)[:,None] & (W_RK != OUT) 
        singles_R = (viableW_RK & (W_RK == IN)).any(1)
        counts_R = jnp.where(singles_R, 1, viable_RK.sum(1))

        return indsW_RK, validW_RK, singles_R, counts_R

    def num_column_children(self, st, c):
        """Compute the number of viable column children"""
        return self._column_options(st, c)[3].sum()

    def _column_child(self, c, st, j, key):
        """the `j`th child (0 <= j < n_child) in order fixed by the frame's key"""
        indsW_RK, validW_RK, singles_R, counts_R = self._column_options(st, c)
        kR, kW = jax.random.split(key)

        perm_R = jax.random.permutation(kR, self.R)
        cnt = counts_R[perm_R] # some of these will be zero
        end = jnp.cumsum(cnt)
        i = jnp.searchsorted(end, j, side="right")
        k = j - (end[i] - cnt[i]) # index within row r
        r = perm_R[i]

        R_new = st.R.at[r].set(IN)

        # Witness order on W(r,c): a function of (key, r) only, so it is the same
        # for every child of r in this frame.
        prio_K = jnp.where(viable_RK[r],
                           jax.random.uniform(jax.random.fold_in(kW, r), (self.K,)),
                           jnp.inf)
        rank_K = jnp.argsort(jnp.argsort(prio_K))
        pick_K = viable_RK[r] & (rank_K == k)
        out_K = viable_RK[r] & (rank_K < k)

        inds = indsW_RK[r]
        W_new = st.W.at[jnp.where(pick_K, inds, self.W)].set(IN, mode="drop")
        W_new = W_new.at[jnp.where(out_K, inds, self.W)].set(OUT, mode="drop")
        W_new = jnp.where(singles_R[r], st.W, W_new)  # row-only child has no witness change

        st = st._replace(cursor_D=st.cursor_D.at[st.top].set(j))

        top = st.top + 1
        n_child = self.num_column_children(st, c)
        return st._replace(
            stack_DR=st.stack_DR.at[top,:].set(R_new),
            stack_DW=st.stack_DW.at[top,:].set(W_new),
            col_D=st.col_D.at[top].set(c),
            n_child_D=st.n_child_D.at[top].set(n_child),
            cursor_D=st.cursor_D.at[top].set(0),
            key_D=st.key_D.at[top].set(key),
            top=top,
        )

    def num_var_children(self, st):
        return (st.und_R.sum() + st.und_W.sum()) * 2

    def _var_child(self, st, j, key):
        """the `j`th child (0 <= j < n_var_child) in order fixed by key"""
        n_undR = st.und_R.sum()
        n_child = self.num_var_children(st)
        perm = jax.random.permutation(key, n_child)
        ch = perm[j]
        val = jnp.where((ch % 2) == 0, jnp.int8(IN), jnp.int8(OUT))
        R_new = jnp.where((ch / 2) < n_undR, val, st.R)
        W_new = jnp.where((ch / 2) >= n_undR, val, st.W)

        top = st.top + 1
        n_child = self.num_var_children(st)

        return st._replace(
            stack_DR=st.stack_DR.at[top,:].set(R_new),
            stack_DW=st.stack_DW.at[top,:].set(W_new),
            col_D=st.col_D.at[top].set(-1),
            n_child_D=st.n_child_D.at[top].set(n_child),
            cursor_D=st.cursor_D.at[top].set(0),
            key_D=st.key_D.at[top].set(key),
            top=top,
        )

    def child(self, c, st):
        j = st.cursor_D[st.top]
        parent_key = st.key_D[st.top]
        child_key = jax.random.fold_in(parent_key, j)
        st = st._replace(cursor_D=st.cursor_D.at[st.top].set(j+1))
        col_child_fn = partial(self.column_child, c)
        return jax.lax.cond(c == -1, self.var_child, col_child_fn, st, j, child_key)


    def pick_uncovered_column(self, st):
        """Return uncovered column with fewest column_children"""
        cand_R = (st.R != OUT)
        avail_W = (st.W != OUT)

        sure_RC = jnp.where(self.validW_RKC, st.in_W[self.indsW_RKC], 0).any(1)
        covered_C = (st.in_R[:,None] & sure_RC).any(0)
        
        nW_RC = jnp.where(self.validW_RKC, avail_W[self.indsW_RKC], 0).any(1)
        per_RC = jnp.where(sure_RC, 1, nW_RC) * cand_R[:,None]
        nchild_C = jnp.where(covered_C, jnp.iinfo(jnp.int64).max, per_RC.sum(0))

        best = jnp.argmin(nchild_C)
        return jnp.where(covered_C.all(), -1, best)

    def search(self, key: PRNGKeyArray) -> State:

        def cond_fn(st):
            return st.status == self.RUNNING

        def can_push(st):
            return (
                (st.top < self.D - 1) 
                & (st.cursor_D[st.top] < st.n_child_D[st.top])
                & ((st.in_R.sum() < self.budgetR) | (st.in_W.sum() < self.budgetW))
                & ((st.und_R.sum() > 0) | (st.und_W.sum() > 0))
            )

        def body_fn(st):
            st = self.propagate(st)
            c = self.pick_uncovered_column(st)
            child_fn = partial(self.child, c)
            pop_fn = lambda x: x._replace(top=x.top-1)
            return jax.lax.cond(can_push, child_fn, pop_fn)

        init_state = State.empty(self.D, self.R, self.W, key)
        return jax.lax.while_loop(cond_fn, body_fn, init_state)


def exact_2dcover(key: PRNGKeyArray, mat_RWC: Int8[Array, "R W C"]) -> tuple[State, Array]:
    """
    Jax implementation of exact 2d cover.  Returns tuple of:
    State, found
    """

