import numpy as np
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

def compact_masked(
	vals: Array,
	mask: Array,
) -> tuple[Array, int]:
	"""
	Compact the vals corresponding to True elements of mask contiguously to
	a result tensor of the same type and shape as vals
	"""
	assert vals.ndim == 1, "only 1D tensors supported"
	assert vals.shape == mask.shape, "vals and mask must have same shape"
	assert mask.dtype == jnp.bool, "mask must be bool dtype"

	N = vals.shape[0]
	inds = jnp.cumsum(mask) - 1
	targ = jnp.where(mask, inds, N)
	dest = jnp.empty((N + 1,), dtype=vals.dtype)
	dest = dest.at[targ].set(vals)
	return dest[:-1], jnp.sum(mask)

def get_max_digits(val, base):
	if val == 0:
		return 1
	return math.ceil(math.log2(val) / math.log2(base))

def tokenize_one_int(
	val: int, 
	base: int,
	zero_token: int, 
	plus_token: int, 
	minus_token: int
) -> np.array:
	abs_val = abs(val)
	D = get_max_digits(abs_val, base)
	powers = base ** np.arange(D - 1, -1, -1)
	sign = np.array(plus_token if val >= 0 else minus_token)
	digits = (abs_val // powers) % base
	return np.concat((sign[None], digits + zero_token))
	

def tokenize_ints(
	vals: Array, 
	vals_mask: Array,
	base: int, 
	zero_token: int, 
	plus_token: int, 
	minus_token: int
) -> tuple[Array, Array]:
	"""
	Converts vals (either int32 or int64 tensor) into a packed Array with:
	SIGN DIGIT{1,} SIGN DIGIT{1,}.

	For example, for base=10, a value of 1983 would be:
	<plus_token> 1 9 8 3

	The output Array is int32 padded with -1 for invalid positions 

	Uses plus_token and minus_token to signify signs.
	Encodes all digits with zero_token offset for value 0
	"""
	assert vals.ndim == 1, "only 1D tensor supported"
	N = vals.shape[0]
	match vals.dtype:
		case jnp.int64:
			D = get_max_digits(2**63, base)
		case jnp.int32:
			D = get_max_digits(2**31, base)
		case _:
			raise RuntimeError(f"only int64 and int32 tensors supported")

	signs = jnp.where(vals >= 0, plus_token, minus_token)
	abs_vals = jnp.abs(vals)
	powers = base ** jnp.arange(D - 1, -1, -1)
	digits = (abs_vals[:, None] // powers[None, :]) % base
	digit_mask = (jnp.cumsum(digits, axis=1) > 0)
	digit_mask = digit_mask.at[:, -1].set(jnp.where(abs_vals == 0, True, digit_mask[:, -1]))
	tokens = jnp.concatenate([signs[:,None], digits + zero_token], axis=1).reshape(-1)
	mask = jnp.concatenate([jnp.ones((N, 1), dtype=bool), digit_mask], axis=1).reshape(-1)
	tokens, _ = compact_masked(tokens, mask)

	vals_mask_expand = jnp.broadcast_to(vals_mask[:,None], (N, D + 1)).reshape(-1)
	tokens_mask, _ = compact_masked(vals_mask_expand, mask)
	return tokens, tokens_mask
