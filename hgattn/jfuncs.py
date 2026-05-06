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

def entropy(vals: jax.Array):
	probs = vals / vals.sum()
	return jnp.where(probs == 0, 0, probs * -jnp.log(probs))

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
	axis: int=0
) -> tuple[Array, int]:
	"""
	Compact the vals corresponding to True elements of mask contiguously along
	`axis` to a result tensor of the same type and shape as vals.

	Return the result, and a number of slices retained 
	"""
	assert mask.dtype == jnp.bool, "mask must be bool dtype"
	assert mask.ndim == 1, "mask must be 1D"
	assert mask.shape[0] == vals.shape[axis], "mask.shape[0] must equal vals.shape[axis]"

	N = vals.shape[axis]
	inds = jnp.cumsum(mask) - 1
	targ = jnp.where(mask, inds, N)
	dest = jnp.empty((N + 1, *vals.shape[1:]), dtype=vals.dtype)
	dest = dest.at[targ].set(vals)
	return dest[:-1], jnp.sum(mask)

def get_max_digits(val, base):
	if val == 0:
		return 1
	D = math.ceil(math.log2(val) / math.log2(base))
	return max(D, 1)

def tokenize_one_int(
	val: int, 
	base: int,
	use_dpse: bool,
) -> tuple[bool, np.array]:
	abs_val = abs(val)
	D = get_max_digits(abs_val, base)
	powers = base ** np.arange(D)
	is_pos = (val >= 0) 
	digits = (abs_val // powers) % base
	if use_dpse:
		digits += np.arange(D) * base
	return is_pos, digits
	

def tokenize_ints(
	vals: Array, 
	vals_mask: Array,
	base: int, 
	use_dpse: bool,
	zero_token: int, 
	plus_token: int, 
	minus_token: int,
	pad_token: int,
) -> tuple[Array, Array]:
	"""
	Converts vals (either int32 or int64 tensor) into a packed Array with:
	SIGN DIGIT{1,} SIGN DIGIT{1,}.

	For example, for base=10, a value of 1983 would be:
	<plus_token> 1 9 8 3

	If `use_dpse` is true, use digit-place-specific encoding: using a separate set of
	`base` tokens for each digit place.

	The output Array is int32 padded with `pad_token` for masked positions 

	Uses plus_token and minus_token to signify signs.
	Encodes all digits with zero_token offset for value 0

	Output:
	A tuple with:
	encoded integers
	corresponding expanded mask
	"""
	assert vals.ndim == 1, "only 1D tensor supported"

	N = vals.shape[0]
	match vals.dtype:
		case jnp.int64:
			D = get_max_digits(2**63, base)
		case jnp.int32:
			D = get_max_digits(2**31, base)
		case _:
			raise RuntimeError(f"only int64 and int32 tensors supported.  got {vals.dtype}")
	
	digit_beg = zero_token
	if use_dpse:
		digit_end = digit_beg + (D * base)
		place_offsets = jnp.arange(D) * base
	else:
		place_offsets = jnp.zeros(D, dtype=jnp.int32)
		digit_end = digit_beg + base

	if plus_token in range(digit_beg, digit_end): 
		raise RuntimeError(f"{plus_token=} overlaps digit range [{digit_beg}, {digit_end})")
	if minus_token in range(digit_beg, digit_end): 
		raise RuntimeError(f"{minus_token=} overlaps digit range [{digit_beg}, {digit_end})")
	if pad_token in range(digit_beg, digit_end): 
		raise RuntimeError(f"{pad_token=} overlaps digit range [{digit_beg}, {digit_end})")
	assert plus_token != minus_token, f"{plus_token=} == {minus_token=}"
	assert plus_token != pad_token, f"{plus_token=} == {pad_token}"

	vals_mask_expand = jnp.broadcast_to(vals_mask[:,None], (N, D + 1)).reshape(-1)
	positions = jnp.cumsum(vals_mask) - 1
	positions = jnp.broadcast_to(positions[:,None], (N, D + 1)).reshape(-1)
	signs = jnp.where(vals >= 0, plus_token, minus_token)
	abs_vals = jnp.abs(vals)
	powers = base ** jnp.arange(D)
	digits = (abs_vals[:, None] // powers[None, :]) % base
	digit_mask = jnp.flip((jnp.cumsum(jnp.flip(digits), axis=1) > 0))
	digit_mask = digit_mask.at[:,0].set(jnp.where(abs_vals == 0, True, digit_mask[:,0]))
	digit_tokens = digits + place_offsets[None,:] + zero_token
	tokens = jnp.concatenate([signs[:,None], digit_tokens], axis=1).reshape(-1)
	mask = jnp.concatenate([jnp.ones((N, 1), dtype=bool), digit_mask], axis=1).reshape(-1)
	mask = jnp.logical_and(mask, vals_mask_expand)
	tokens, ntoks = compact_masked(tokens, mask)
	O = tokens.shape[0]
	tokens = jnp.where(jnp.arange(O) < ntoks, tokens, pad_token)
	source_positions, _ = compact_masked(positions, mask)
	source_positions = jnp.where(jnp.arange(O) < ntoks, source_positions, -1)
	return tokens, source_positions 


def masked_arange(mask):
	positions = jnp.cumsum(mask) - 1
	O = positions.shape[0]
	source_positions, ntoks = compact_masked(positions, mask)
	source_positions = jnp.where(jnp.arange(O) < ntoks, source_positions, -1)
	return source_positions

def mix_bits32(h):
    # Ensure we are working with unsigned 32-bit integers
    h = h.astype(jnp.uint32)
    
    # The constants must also be uint32
    c1 = jnp.uint32(0x85ebca6b)
    c2 = jnp.uint32(0xc2b2ae35)
    
    h ^= h >> 16
    h *= c1
    h ^= h >> 13
    h *= c2
    h ^= h >> 16
    return h

def mix_bits64(h):
    # Ensure unsigned 64-bit math
    h = h.astype(jnp.uint64)
    
    # 64-bit constants (SplitMix64 / MurmurHash3 constants)
    c1 = jnp.uint64(0xff51afd7ed558ccd)
    c2 = jnp.uint64(0xc4ceb9fe1a85ec53)
    
    h ^= h >> 33
    h *= c1
    h ^= h >> 33
    h *= c2
    h ^= h >> 33
    return h

def hash(vals):
	if vals.itemsize == 4:
		return jnp.bitwise_xor.reduce(jax.vmap(mix_bits32)(vals.flatten()))
	elif vals.itemsize == 8:
		return jnp.bitwise_xor.reduce(jax.vmap(mix_bits64)(vals.flatten()))
	else:
		raise TypeError(f"Unsupported bit-width: {vals.itemsize * 8}-bit")

