"""
Jax implementation of parallel exact cover

Exact cover starts with a 0-1 matrix mat_RC of R rows by C columns.  A solution
consists of a 0-1 chosen_R identifying a subset of rows such that every column is
covered by exactly one row.  That is:

jnp.all((mat_RC * chosen_R[:,None]).sum(axis=0) == 1)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Int, Array, Bool, PRNGKeyArray
from typing import NamedTuple

def gen_mat(key: PRNGKeyArray, R: int, C: int, avg_col_weight: float) -> Array:
	return jax.random.bernoulli(key, avg_col_weight, (R, C)).astype(jnp.int32)

class State(NamedTuple):
	status: Int[Array, ""]
	d: Int[Array, ""]
	col_D: Int[Array, "depth"]
	row_D: Int[Array, "depth"]
	descend: Bool[Array, ""]
	block_R: Int[Array, "row"]
	avail_C: Int[Array, "col"]
	chosen_R: Int[Array, "row"]
	start: Int[Array, ""]
	proceed: Bool[Array, ""]
	c: Int[Array, ""]
	iters: Int[Array, ""]

def _exact_cover(mat_RC: Int[Array, "R C"]) -> tuple[State, Array]:
	"""
	Jax implementation of exact cover.  Returns tuple of:
	State, found
	"""
	RUNNING, FOUND, INFEASIBLE = 0, 1, 2 # loop status
	R, C = mat_RC.shape
	clash_RR = (jnp.einsum('rc, sc -> rs', mat_RC, mat_RC) > 0).astype(jnp.int32)

	init_state = State(
		status = jnp.int32(RUNNING),
		d = jnp.int32(0),
		col_D = jnp.empty(C, jnp.int32),
		row_D = jnp.empty(C, jnp.int32),
		descend = jnp.bool(True),
		block_R = jnp.zeros(R, jnp.int32),
		avail_C = jnp.ones(C, jnp.int32),
		chosen_R = jnp.zeros(R, jnp.int32),
		start = jnp.int32(0),
		proceed = jnp.bool(True),
		c = jnp.int32(0),
		iters = jnp.int32(0),
	)

	def body_fn(state: State) -> State:
		state = state._replace(iters=state.iters+1)

		def descend(st):
			st = st._replace(status=jnp.where(st.avail_C.sum() == 0, FOUND, st.status))
			compat_RC = mat_RC * (st.block_R[:,None] == 0)
			num_compat_C = jnp.where(st.avail_C, compat_RC.sum(0), R + 1)

			def bail():
				return st._replace(descend=False)
			def begin():
				c = jnp.argmin(num_compat_C)
				return st._replace(col_D=st.col_D.at[st.d].set(c), start=0, c=c)
			return jax.lax.cond(jnp.any(num_compat_C == 0), bail, begin)

		def stay(st):
			def fail():
				return st._replace(status=INFEASIBLE)
			def step():
				pd = st.d - 1
				r = st.row_D[pd]
				return st._replace(
					d = pd,
					chosen_R = st.chosen_R.at[r].set(0),
					block_R = st.block_R - clash_RR[r],
					avail_C = st.avail_C + mat_RC[r,:],
					start = r + 1,
					c = st.col_D[pd],
				)
			return jax.lax.cond(st.d == 0, fail, step)

		def shared(st):
			compat_RC = mat_RC * (st.block_R[:,None] == 0)
			cands_R = jnp.where(jnp.arange(R) < st.start, 0, compat_RC[:,st.c])

			def br1():
				return st._replace(descend=False)

			def br2():
				r = jnp.argmax(cands_R)
				return st._replace(
					d = st.d + 1,
					row_D = st.row_D.at[st.d].set(r),
					descend = True,
					block_R = st.block_R + clash_RR[r],
					avail_C = st.avail_C - mat_RC[r,:],
					chosen_R = st.chosen_R.at[r].set(1),
				)

			return jax.lax.cond(cands_R.sum() == 0, br1, br2)

		state = jax.lax.cond(state.descend, descend, stay, state)
		state = jax.lax.cond(state.proceed, shared, lambda s: s, state)
		return state

	def cond_fn(state: State) -> bool:
		return state.status == jnp.int32(RUNNING)

	final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
	return final_state, (final_state.status == jnp.int32(FOUND))

class ExactCoverResult(NamedTuple):
	chosen_R: Int[Array, "row"]
	found: Bool[Array, ""]

class Evaluation(NamedTuple):
	passed: Int[Array, ""]
	failed: Int[Array, ""]
	skipped: Int[Array, ""]

@jax.jit
def exact_cover(mat_BRC: Array) -> tuple[Array, Array]:
	return jax.vmap(_exact_cover)(mat_BRC)

def validate(mat_RC: Array, chosen_R: Array) -> Array:
	return jnp.all((mat_RC * chosen_R[:,None]).sum(0) == 1)

def get_lowered(mat_RC):
	return jax.jit(exact_cover).lower(mat_RC).compile().as_text()


def test(seed: int, R: int, C: int, avg_col_weight: float, reps: int, batch_size: int) -> dict:
	stats = dict(passed=0, failed=0, skipped=0)
	test_key = jax.random.key(seed)
	def gen_mat_wrap(key):
		return gen_mat(key, R, C, avg_col_weight)

	for r in range(0, reps, batch_size):
		print(f"step {r}")
		rep_key_B = jax.random.split(jax.random.fold_in(test_key, r), num=batch_size)
		mat_BRC = jax.vmap(gen_mat_wrap)(rep_key_B)
		state, found = exact_cover(mat_BRC)
		valid_B = jax.vmap(validate)(mat_BRC, state.chosen_R)
		num_passed = (valid_B & found).sum()
		num_failed = (~valid_B & found).sum()
		num_skipped = (~found).sum()

		stats['passed'] += num_passed.item()
		stats['failed'] += num_failed.item()
		stats['skipped'] += num_skipped.item()

	return stats

