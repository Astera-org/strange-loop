import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, Array
from .types import TokensAndProbs 
from dataclasses import dataclass

"""
A dataset of patterns: A B S A B S A B S ...
in which S = (A + B) % mod_val
where A and B are drawn from [0, vocab_size) and mod_val is in [1, vocab_size) 

The target positions are defined to be just the S role.

The train / test split is a random sampling of possible (A, B) pairs, with
train_split_frac of them given to the train_split.
"""


@dataclass
class ModAdditionOpts:
	context_len: int
	vocab_size: int
	mod_val: int
	train_split_frac: float  # fraction of possible (A, B) values given to train split

class ModAdditionDataset(eqx.Module):
	opts: ModAdditionOpts = eqx.field(static=True)
	coords: jax.Array
	distA: jax.Array
	distBgivenA: jax.Array

	def __init__(self, opts: ModAdditionOpts, is_train: bool, seed: int):
		self.opts = opts
		V = opts.vocab_size
		inds = jnp.stack((jnp.arange(V * V) // V, jnp.arange(V * V) % V), axis=1)
		key = jax.random.key(seed)
		inds = jax.random.permutation(key, inds)
		split = int(V * V * opts.train_split_frac) 
		if is_train:
			self.coords = inds[:split]
		else:
			self.coords = inds[split:]
		distA = jnp.zeros(V).at[self.coords[:,0]].add(1.0) 
		self.distA = distA / distA.sum()
		distAB = jnp.zeros((V, V)).at[self.coords[:,0], self.coords[:,1]].add(1.0)
		self.distBgivenA = distAB / distA[:,None]

	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: Key) -> TokensAndProbs:
		true_val, false_val = jnp.array(True), jnp.array(False)
		
		def b0_A(key, state, arg):
			a, b = jax.random.choice(key, self.coords)
			return jnp.array([1, a, b]), (a, self.distA, true_val, false_val)
	
		def b1_B(key, state, arg):
			a, b = arg[:2]
			s = (a + b) % self.opts.mod_val 
			return jnp.array([2, s, -1]), (b, self.distBgivenA[a,:], true_val, false_val)

		def b2_S(key, state, arg):
			s = arg[0]
			distS = jax.nn.one_hot(s, self.opts.mod_val)
			return jnp.array([0, -1, -1]), (s, distS, true_val, true_val) 

		branches = b0_A, b1_B, b2_S
		
		def scan_fn(carry: Int[Array, "slots"], key: Key[Array, ""]) -> tuple:
			state, arg = carry[0], carry[1:]
			return jax.lax.switch(state, branches, key, state, arg)

		def single_scan(key, length):
			init = jnp.array([0, -1, -1])
			keys = jax.random.split(key, num=length)
			return jax.lax.scan(scan_fn, init, keys)

		batch_scan = jax.vmap(single_scan, in_axes=(0, None))
		_, content = batch_scan(key_B, self.opts.context_len)

		return TokensAndProbs(jax.random.key_data(key_B), *content)

