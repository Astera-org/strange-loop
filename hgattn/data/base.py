from dataclasses import replace
from abc import abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp
from .types import TokensAndProbs
from jaxtyping import Array, Int, PRNGKeyArray
from .. import jfuncs

class BaseDataset(eqx.Module):
	is_train: bool = eqx.field(static=True)
	train_frac: float = eqx.field(static=True)

	def __init__(self, is_train: bool, train_frac: float):
		self.is_train = is_train
		self.train_frac = train_frac

	@abstractmethod
	def _generate_one(self, key: PRNGKeyArray) -> TokensAndProbs:
		"""
		Core definition of the generator.  
		"""
		...

	@abstractmethod
	def validate(self, tokens: Array) -> Int[Array, ""]:
		"""
		Accepts the tokens, and returns a scalar integer status array
		"""
		...

	@abstractmethod
	def print_raw_item(self, item: TokensAndProbs) -> str:
		"""
		Produces a human-readable rendering for the item
		"""
		...

	@eqx.filter_jit
	def _gen_batch(self, key_B: PRNGKeyArray) -> TokensAndProbs:
		"""
		Applies batching, rejection sampling and splitting of the
		underlying _generate_one
		"""
		item = eqx.filter_vmap(self._generate_one)(key_B)
		B = key_B.shape[0]
		train_size = int(B * self.train_frac)

		is_train_frac = (item.split_hash % 1048576) < int(self.train_frac * 1048576)
		is_active = ((self.is_train == is_train_frac) & item.active)
		size = train_size if self.is_train else B - train_size

		item = replace(item, active=is_active)

		def _fraction(x):
			x, _ = jfuncs.partition_masked(x, is_active, 0)
			return x[:size]

		return jax.tree.map(_fraction, item)

	@eqx.filter_jit
	def _validate_item(self, item: TokensAndProbs) -> Array:
		return eqx.filter_vmap(self.validate)(item)

	def validate_item(self, item: TokensAndProbs) -> tuple[bool, list[int]]:
		status_B = self._validate_item(item)
		pct_active = item.active.sum() / item.active.shape[0]
		if pct_active < 0.5:
			return False, (
					f"Item had only {pct_active} active elements")

		is_valid_B = jnp.where(item.active, status_B == 0, True)
		all_passed = jnp.all(is_valid_B)

		return all_passed.item(), status_B.tolist()


