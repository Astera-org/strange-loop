import jax.numpy as jnp
import jax
from jaxtyping import Shaped, Array, Int, Bool, Scalar
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
