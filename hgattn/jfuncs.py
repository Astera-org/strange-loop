import jax.numpy as jnp
import jax
import math
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

def range_mask(beg: Array, end: Array, length: int) -> Array:
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

def first_index_of(vals: Array, val: Any) -> int:
	n = vals.shape[0]
	mask = vals == val
	inds = jnp.where(mask, jnp.arange(n), n)
	idx = jnp.argmin(inds)
	return jnp.where(mask.any(), idx, -1)

def tokenize_ints(
	vals: Array, 
	base: int, 
	digit_zero: int, 
	plus_token: int, 
	minus_token: int
) -> Array:
	"""
	Converts vals (either int32 or int64 tensor) into a packed Array with:
	SIGN DIGIT{1,} SIGN DIGIT{1,}.

	For example, for base=10, a value of 1983 would be:
	<plus_token> 1 9 8 3

	The output Array is int32 padded with -1 for invalid positions 

	Uses plus_token and minus_token to signify signs.
	Encodes all digits with digit_zero offset for value 0
	"""
	assert vals.ndim == 1, "only 1D tensor supported"
	N = vals.shape[0]
	match vals.dtype:
		# TODO: Check this math
		case jnp.int64:
			max_digits = math.floor(63 / math.log2(base))
		case jnp.int32:
			max_digits = math.floor(31 / math.log2(base))
		case _:
			raise RuntimeError(f"only int64 and int32 tensors supported")

	signs = jnp.where(vals >= 0, plus_token, minus_token)
	abs_vals = jnp.abs(vals)
	powers = base ** jnp.arange(max_digits - 1, -1, -1)
	digits = (abs_vals[:, None] // powers[None, :]) % base + digit_zero
	digit_mask = (jnp.cumsum(digits, axis=1) > 0)
	digit_mask = digit_mask.at[:, -1].set(jnp.where(abs_vals == 0, True, digit_mask[:, -1]))
	tokens = jnp.concatenate([signs[:,None], digits], axis=1).reshape(-1)
	mask = jnp.concatenate([jnp.ones((N, 1), dtype=bool), digit_mask], axis=1).reshape(-1)
	indices = jnp.cumsum(mask) - 1
	out_size = N * (max_digits + 1)
	out = jnp.full((out_size + 1,), -1, dtype=jnp.int32)
	targets = jnp.where(mask, indices, out_size)
	res = out.at[targets].set(tokens)
	return res[:out_size]

