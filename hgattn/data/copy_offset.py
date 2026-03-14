import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Int, Array, PRNGKeyArray
from dataclasses import dataclass
from .. import jfuncs
from .types import TokensAndProbs
import math

@dataclass
class CopyOffsetOpts:
	context_len: int
	num_vals: int
	op_frequency: float
	fixed_offsets: list[int]|None # 

	def __post_init__(self):
		if self.fixed_offsets is None:
			self.fixed_offsets = list(range(self.num_vals))

		if not all(0 <= f < self.num_vals for f in self.fixed_offsets):
			raise RuntimeError(
				f"fixed offsets must all be in [0, num_vals). "
				f"Got {self.fixed_offsets} for num_vals={self.num_vals}")


class CopyOffsetDataset(eqx.Module):
	opts: CopyOffsetOpts = eqx.field(static=True) 
	fixed_offsets: Int[Array, "offset"]

	def __init__(self, opts: CopyOffsetOpts):
		self.opts = opts
		self.fixed_offsets = jnp.array(opts.fixed_offsets)

	@property
	def vocab_size(self):
		return self.opts.num_vals + 1 # values + OP

	@eqx.filter_jit
	def _gen_item(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		B = key_B.shape[0]
		C = self.opts.context_len
		V = self.opts.num_vals
		op_token = V
		I = math.ceil(1 / self.opts.op_frequency)
		op_freq = self.opts.op_frequency
		plain = jnp.ones((V,)) * (1.0 - op_freq) 
		
		non_target_prob = jnp.concat((plain, jnp.array([op_freq])))

		# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
		def scan_fn(carry, _: Any) -> tuple:
			"""
			dists is int[L], distances to special tokens
			write is int[L], tokens to write
			is_target is bool[L], whether tokens are targets

			"""
			dists, write, targets, key = carry
			dists = dists - 1
			k1, k2, k3, next_key = jax.random.split(key, 4)
			slot, found = jfuncs.find_first_value(dists, 0)
			token = jnp.where(found, write[slot], jax.random.randint(k1, (), 0, V))

			is_target = jnp.where(found, targets[slot], jnp.array(False))

			is_source = jax.random.choice(k2, I) == 0
			is_source = jnp.logical_and(is_source, jnp.logical_not(found))

			offset = jax.random.choice(k3, self.fixed_offsets)
			new_dists = jnp.arange(3) + offset
			occupied = jnp.any(jnp.isin(new_dists, dists))

			use_source = jnp.logical_and(is_source, jnp.logical_not(occupied))

			_, free_inds = jax.lax.top_k(dists < 0, k=3)

			cur_dists = dists[free_inds]
			dists = dists.at[free_inds].set(jnp.where(use_source, new_dists, cur_dists))

			cur_write = write[free_inds]
			new_write = jnp.array([offset, op_token, token])
			write = write.at[free_inds].set(jnp.where(use_source, new_write, cur_write))

			cur_targets = targets[free_inds]
			new_targets = jnp.array([False, False, True])
			targets = targets.at[free_inds].set(jnp.where(use_source, new_targets, cur_targets))

			probs = jnp.where(is_target, jax.nn.one_hot(token, V+1), non_target_prob)

			input_mask = jnp.array(True)

			target_mask = is_target

			carry = dists, write, targets, next_key
			return carry, (token, probs, input_mask, target_mask) 

		def single_scan(state, length):
			return jax.lax.scan(scan_fn, state, length=length)

		dists = jnp.zeros((B, 30), dtype=jnp.int32)
		write = jnp.zeros((B, 30), dtype=jnp.int32)
		is_target = jnp.full((B, 30), False)
		carry = dists, write, is_target, key_B
		batch_scan = jax.vmap(single_scan, in_axes=(0, None))
		_, content = batch_scan(carry, self.opts.context_len)
		return TokensAndProbs(jax.random.key_data(key_B), *content)

