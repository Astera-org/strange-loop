import jax
import jax.numpy as jnp
from jaxtyping import Int, Int8, Array, PRNGKeyArray
from typing import NamedTuple


class State(NamedTuple):
    status: Int8[Array, ""]
    stack_DR: Int8[Array, "D R"]
    stack_DW: Int8[Array, "D W"]
    col_D: Int[Array, "D"]
    var_D: Int[Array, "D"]
    n_child_D: Int[Array, "D"]  # number of children
    cursor_D: Int[Array, "D"]   # next child to try
    top: Int8

	@property
	def in_R(self):
		return self.stack_DR[self.top,:] == self.IN

	@property
	def out_R(self):
		return self.stack_DR[self.top,:] == self.OUT

	@property
	def und_R(self):
        return self.stack_DR[self.top,:] == self.UND

	@property
	def in_W(self):
        return self.stack_DW[self.top,:] == self.IN

	@property
	def out_W(self):
        return self.stack_DW[self.top,:] == self.OUT

	@property
	def und_W(self):
        return self.stack_DW[self.top,:] == self.UND

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
    RUNNING = jnp.int8(0)
	FOUND = jnp.int8(1)
	INFEASIBLE = jnp.int8(2)

    IN = jnp.int8(0)
    OUT = jnp.int8(1)
    UND = jnp.int8(2)

    def __init__(
        self,
        graph_RWC: Int8[Array, "R W C"],
        budgetR: int,
        budgetW: int,
    ):
        self.indsW_RKC, self.validW_RKC = compress_graph(graph_RWC) 
        self.budgetR = budgetR
        self.budgetW = budgetW

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
                self.INFEASIBLE,
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
                self.INFEASIBLE,
                self.RUNNING,
                status
            )

            more = ((out_R | in_R).any() | (out_W | in_W).any()) & (status == self.RUNNING)

            st = st._replace(
                status=status,
                R=R.at[in_R].set(self.IN).at[out_R].set(self.OUT),
                W=W.at[in_W].set(self.IN).at[out_W].set(self.OUT),
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

    def column_child(
        self,
        st: State,
        c: Int[Array, ""],
        kth: Int[Array, ""],
    ) -> State:
        """
        Produce the `kth` column child from `st` for column 'c'
        """
        indsW_RK = self.indsW_RKC[:,:,c]
        validW_RK = self.validW_RKC[:,:,c]








def exact_2dcover(key: PRNGKeyArray, mat_RWC: Int8[Array, "R W C"]) -> tuple[State, Array]:
    """
    Jax implementation of exact 2d cover.  Returns tuple of:
    State, found
    """

