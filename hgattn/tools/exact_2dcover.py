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

class ExactCover2D:
    def __init__(
        self,
        graph_RWC: Int8[Array, "R W C"],
        budgetR: int,
        budgetW: int,
    ):
        self.graph_RWC = graph_RWC
        self.budgetR = budgetR
        self.budgetW = budgetW

    def propagate(state: State) -> State:

        def body_fn(st):
            sure_RC = (self.graph_RWC & st.in_W[None,:,None]).any(1)
            poss_RC = (self.graph_RWC & ~st.out_W[None,:,None]).any(1)
            cover_C = (st.in_R[:,None] & sure_RC).sum(0)
            cand_C = (~st.out_R[:,None] & poss_RC).sum(0)
            if (cover_C >= 2).any() or (cand_C == 0).any():
                st = st._replace(status=INFEASIBLE)
            covered_C = (cover_C == 1)

            out_R = st.und_R & (sure_RC & covered_C[None,:]).any(1)

            # witnesses whose addition would double-cover some column
            newcov_WC = (st.in_R[:,None,None] & (sure_RC[:,None,:] | self.graph_RWC)).sum(0)
            out_W = st.und_W & (newcov_WC >= 2).any(1)

            in_R, in_W = self.unit_rules(st, poss_RC, cand_C, covered_C)

            if (in_R & out_R).any() or (in_W & out_W).any():
                st = st._replace(status=INFEASIBLE)

            if not (out_R | in_R).any() and not (out_W | in_W).any():
                return st

            st = st.apply(in_R, out_R, in_W, out_W)

        def cond_fn(st):
            pass

        state = jax.lax.while_loop(cond_fn, body_fn, state)
        return state 


def exact_2dcover(key: PRNGKeyArray, mat_RWC: Int8[Array, "R W C"]) -> tuple[State, Array]:
    """
    Jax implementation of exact 2d cover.  Returns tuple of:
    State, found
    """

