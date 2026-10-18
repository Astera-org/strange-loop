from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import numpy as np
from ortools.sat.python import cp_model
from dataclasses import dataclass
from enum import StrEnum, auto
from . import DatasetOpts
from ..utils import MirroredEnum
from .. import jfuncs

class Token(StrEnum):
    S = auto()
    P = auto()
    Q = auto()

class Task(MirroredEnum):
	THREE_EDGE = "three-edge"   # [S-P Bag] [Q Q Q ...] = [S S S ...]

@dataclass
class ThreeEdgeOpts(DatasetOpts):
    task: Task
    num_pq: int
    num_s: int
    num_nz: int
    x_rate: float


class ThreeEdgeDataset(eqx.Module):
    opts: ThreeEdgeOpts = eqx.field(static=True)
    num_tokens: dict[Token, int] = eqx.field(static=True)
    graph: Array

    def __init__(self, opts: ThreeEdgeOpts, is_train: bool, seed: int):
        self.opts = opts
        key = jax.random.key(seed)
        I, K = opts.num_pq, opts.num_s
        self.num_tokens = { Token.S: K, Token.P: I, Token.Q: I }
        choice_fn = lambda key: jax.random.choice(key, K, (opts.num_nz,), replace=False)
        i, j, _ = jnp.ogrid[:I,:I,:1]
        inds = jax.vmap(choice_fn)(jax.random.split(key, num=I*I)).reshape(I, I, -1)
        self.graph = jnp.zeros((I, I, K), dtype=jnp.int32).at[i, j, inds].set(1)

    def _maybe_solve(self, key: PRNGKeyArray) -> tuple[Array, Array, Array]:
        """
        Returns a representation of triplets (p, q, s) sorted by p in [0, I)
        q_I: the q value for triplet i
        s_I: the s value for triplet i 
        x_K: bool[K]: set of s present in the solution 
        feasible: bool[], whether the solution is feasible.  (if false, do not use)
        """
        # return w, x, feasible
        I = self.opts.num_pq
        J = self.opts.num_pq
        K = self.opts.num_s
        xkey, wkey = jax.random.split(key)
        x_K = jax.random.bernoulli(xkey, self.opts.x_rate, K)
        marg_IJ = jnp.einsum('ijk, k -> ij', self.graph, x_K.astype(jnp.int32))
        marg_I = marg_IJ.sum(axis=1)

        # (i,j) choices with a unique k for the particular x_K
        valid_IJ = (marg_IJ == 1).astype(jnp.int32) 

        # i has at least one j which forms a valid (i,j) pair
        valid_I = valid_IJ.sum(axis=1)
        feasible = jnp.all(valid_I >= 1)
        cdf_IJ = jnp.cumsum(valid_IJ, axis=1) - 1
        target_I = jax.random.randint(wkey, shape=(I,), minval=0, maxval=valid_I)
        p_I = jnp.arange(I)
        q_I = jax.vmap(partial(jnp.searchsorted, side="left"))(cdf_IJ, target_I)
        chosen_IK = jnp.where(x_K[None,:], self.graph[p_I, q_I], 0)
        # chosen_IK, num_s = jfuncs.partition_masked(self.graph[p_I, q_I], x_K, axis=1)
        s_I = jnp.argmax(chosen_IK, 1) # 

        return q_I, s_I, x_K, feasible

    def _validate_solution(self, q_I, s_I, x_K) -> Array:
        I = self.opts.num_pq
        J = self.opts.num_pq
        K = self.opts.num_s
        s_valid = jnp.all(x_K[s_I])
        p_I = jnp.arange(I)
        chosen_IK = jnp.where(x_K[None,:], self.graph[p_I, q_I], 0)
        chosen_I = chosen_IK.sum(axis=1)
        qs_valid = jnp.all(chosen_I == 1)
        return s_valid & qs_valid 

    def _generate_three_edge(self, key: PRNGKeyArray) -> tuple[Array]:
        """
        Generate the content of one item 
        obs_sym: [S-P Bag] [Q Q Q ...] = [S S S ...] EOS
        input_mask: bool array
        target_code: identifies r.h.s. tokens
        split_hash: hash for splitting into train / test
        """
        I = self.opts.num_pq
        J = self.opts.num_pq
        K = self.opts.num_s
        L = K + 3 * I + 2
        keys = jax.random.split(key, num=3)
        q_I, s_I, x_K, feasible = self._maybe_solve(keys[0])
        q_perm_I = jax.random.permutation(keys[1], I)
        s_K, num_s = jfuncs.partition_masked(jnp.arange(K), x_K)
        obs_sym = jnp.empty(L, dtype=jnp.int32)
        # obs_sym = obs_sym.at[


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=180)
    opts = ThreeEdgeOpts("three-edge", 10, 50, 2, 0.5)
    ds = ThreeEdgeDataset(opts, True, 12345)
    key = jax.random.key(93848)
    q_I, s_I, x_K, feasible = ds._maybe_solve(key)
    is_valid = ds._validate_solution(q_I, s_I, x_K)




