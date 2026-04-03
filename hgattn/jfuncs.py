import jax.numpy as jnp
import jax
from jaxtyping import Shaped, Array, Int, Bool, Scalar, Key
from typing import Any

def find_first_value(
	target: Shaped[Array, '...'], 
	value: Scalar|Any,
) -> tuple[Int[Array, ""], Bool[Array, ""]]:
	mask = (target == value)
	exists = jnp.any(mask)
	index = jnp.argmax(mask)
	return index, exists

def range_mask(beg: int, end: int, length: int) -> Array:
	inds = jnp.arange(length)
	return jnp.logical_and(inds >= beg, inds < end)

def log_range_mask(beg: int, end: int, length: int) -> Array:
	mask = range_mask(beg, end, length)
	return jnp.where(mask, 0, -jnp.inf)


def geometric_logpmf(p: float, end: int):
	"""
	Return a portion of a geometric distribution at k = [1, end + 1)
	"""
	return jnp.arange(end) * jnp.log1p(-p) + jnp.log(p)

def subsample_mask(key: Key, mask: jax.Array, p: float) -> Array:
	k = mask.sum()
	n_select = jnp.floor(k * p).astype(jnp.int32)

	rand = jax.random.uniform(key, shape=mask.shape)
	rand = jnp.where(mask, rand, 2.0)
	order = jnp.argsort(rand)
	ranks = jnp.zeros_like(order)
	ranks = ranks.at[order].set(jnp.arange(mask.size))
	return ranks < n_select

