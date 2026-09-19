import math
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax
from jaxtyping import Array


def is_prime(p: int) -> bool:
	if p <= 1:
		return False
	if p <= 3:
		return True
	if p % 2 == 0 or p % 3 == 0:
		return False
	
	limit = math.isqrt(p)
	for factor in range(5, limit + 1, 6):
		if p % factor == 0 or p % (factor + 2) == 0:
			return False
	return True


def inv_mod(d: np.ndarray, p: int):
	"""
	Compute modular multiplicative inverse 
	x s.t. dx mod p = 1.
	This relies on Fermat's little theorem:  d^p = d (mod p)
	Therefore:
	d^{p-1} = 1 (mod p) 
	d^{p-2} = d^{-1} (mod p)
	So, this function computes d^{p-2}
	"""
	if not is_prime(p):
		raise RuntimeError(f"inv_mod only works if p is prime.  Received {p=}")

	result = np.full((d.shape[0],), 1, dtype=np.int32) 
	base = d % p
	e = p - 2 

	while e:
		if e & 1:
			result = (result * base) % p
		base = (base * base) % p
		e >>= 1
	return result

@dataclass
class GaussElimResultProto:
	mat: np.ndarray          # [M | y] augmented reduced matrix
	mat_raw: np.ndarray      # [M | y] non-modular equivalent
	free_cols: np.ndarray    # free_cols[k] = True if column k is free
	pivot_rows: np.ndarray   # pivot_rows[k] = r 
	consistent: bool
	mod_val: int

	def __repr__(self):
		return "\n".join((f"{k}:\n{v}\n" for k, v in self.__dict__.items()))

	def _solve(self):
		ncols = self.mat.shape[1] - 1
		d = self.mat[self.pivot_rows,np.arange(ncols)]
		y = self.mat[:,ncols]
		d_inv = inv_mod(d, self.mod_val)
		x = (y * d_inv) % self.mod_val
		return x

	def solve(self):
		return self._solve()

	def solve_raw(self):
		vals = self._solve()
		return np.where(vals > self.mod_val // 2, vals - self.mod_val, vals)


def gauss_elimination_proto(
	mat: np.ndarray, 
	y: np.ndarray,
	mod_val: int,
) -> GaussElimResult:
	"""
	Performs gaussian elimination on mat, producing the result, which can
	further be used to generate solutions if they exist.
	"""
	nrows, ncols = mat.shape
	assert y.shape[0] == nrows, f"y must have same length as columns of m"

	free_cols = np.full((ncols,), True)
	used_rows = np.full((nrows,), False)
	pivot_rows = np.full((ncols,), -1, dtype=np.int32)

	aug = np.concat((mat, y[:,None]), axis=1)
	# ones = np.ones(nrows, dtype=np.int32)
	# prev_pivot = 1

	for k in range(ncols):
		r = np.argmax(~used_rows & (aug[:,k] != 0))
		pivot = aug[r,k]
		if not used_rows[r] and pivot != 0:
			row = aug[r:r+1,:]
			col = aug[:,k:k+1].copy()
			col[r,:] = 0
			# divis = np.where(np.arange(nrows) == r, ones, prev_pivot)
			# aug = ((aug * pivot - row * col) // divis[:,None]) % mod_val
			aug = (aug * pivot - row * col) % mod_val
			free_cols[k] = False
			used_rows[r] = True
			pivot_rows[k] = r
			# prev_pivot = pivot

	consistent = np.all(used_rows | (aug[:,ncols] == 0))
	aug_raw = np.where(aug > mod_val // 2, aug - mod_val, aug)
	return GaussElimResult(aug, aug_raw, free_cols, pivot_rows, consistent, mod_val)

def gauss_elimination(
	mat: Array,
	y: Array,
	mod_val: int,
) -> GaussElimResult:
	"""
	Perform Gauss-Jordan elimination on the [mat | y] system.
	"""
	nrows, ncols = mat.shape
	assert y.shape[0] == nrows, f"y must have the same length as columns of m"

	free_cols = jnp.full((ncols,), True)
	used_rows = jnp.full((nrows,), False)
	pivot_rows = jnp.full((ncols,), -1, dtype=jnp.int32)

	aug = jnp.concatenate((mat, y[:,None]), axis=1)

	def update_fn(aug, free_cols, used_rows, pivot_rows, r, k):
		pivot = aug[r, k]
		row = aug[r, :][None, :]
		col = aug[:, k][:, None]
		col = col.at[r, :].set(0)
		aug = (aug * pivot - row * col) % mod_val
		free_cols = free_cols.at[k].set(False)
		used_rows = used_rows.at[r].set(True)
		pivot_rows = pivot_rows.at[k].set(r)
		return aug, free_cols, used_rows, pivot_rows 

	def step_fn(k, carry):
		aug, free_cols, used_rows, pivot_rows = carry
		r = jnp.argmax(~used_rows & (aug[:, k] != 0))
		pivot = aug[r,k]
		do_update = ~used_rows[r] & (pivot != 0)
		return jax.lax.cond(do_update, update_fn, lambda *x: x[:4], *carry, r, k)

	init_vals = aug, free_cols, used_rows, pivot_rows

	out = jax.lax.fori_loop(0, ncols, step_fn, init_vals) 
	aug, free_cols, used_rows, pivot_rows = out
	consistent = jnp.all(used_rows | (aug[:,ncols] == 0))
	aug_raw = jnp.where(aug > mod_val // 2, aug - mod_val, aug)
	return GaussElimResult(aug, aug_raw, free_cols, pivot_rows, consistent, mod_val)


