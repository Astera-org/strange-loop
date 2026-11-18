import numpy as np
from numpy.random import Generator

def gen_witness_graph(
	R: int, 
	W: int, 
	C: int, 
	n_witness_per_link: int, 
	rng: Generator,
) -> np.array:
	graph = np.zeros((R, W, C), dtype=np.int32)
	rand_vals = rng.random((R, W, C))
	topk_RWC = np.argpartition(rand_vals, n_witness_per_link, axis=1)
	k_inds = topk_RWC[:,:n_witness_per_link,:]
	graph[np.arange(R)[:,None,None], k_inds, np.arange(C)[None,None,:]] = 1
	return graph

def simple_graph(
	R: int, 
	W: int, 
	C: int, 
	cell_prob: float,
	rng: Generator,
) -> np.array:
	return (rng.uniform(0.0, 1.0, (R, W, C)) < cell_prob).astype(np.int32)


def gen_factored_witness_graph(
	R: int, 
	W: int, 
	C: int, 
	witness_per_row: int,
	witness_per_col: int,
	rng: Generator
) -> np.array:
	"""
	Generate a factored graph expressed as:
	fac1_RW[:,:,None] * fac2_WC[None,:,:]
	"""
	fac1_RW = np.zeros((R, W), dtype=np.int32)
	fac2_WC = np.zeros((W, C), dtype=np.int32)
	rand1 = rng.random((R, W))
	inds_RW = np.argpartition(rand1, witness_per_row, axis=1)
	inds_RK = inds_RW[:,:witness_per_row]

	rand2 = rng.random((W, C))
	inds_WC = np.argpartition(rand2, witness_per_col, axis=0)
	inds_KC = inds_WC[:witness_per_col,:]

	fac1_RW[np.arange(R)[:,None],inds_RK] = 1
	fac2_WC[inds_KC,np.arange(C)[None,:]] = 1

	return fac1_RW[:,:,None] * fac2_WC[None,:,:]


# leftover from three-edge
"""
def _init_factored_graph(self, key: PRNGKeyArray):
	W = self.opts.n_vocab_witness
	Q = self.opts.n_vocab_query
	T = self.opts.n_vocab_target
	d = self.opts.graph_density

	# must be exactly one target per witness
	t_density = 1 / T 
	q_density = d
	qk, tk = jax.random.split(key)

	def _density_mask(key, density, shape):
		k1, k2, k3 = jax.random.split(key, num=3)
		n = np.prod(shape)
		mask = jax.random.permutation(k1, jnp.arange(n) < jnp.round(n * density))
		mat = mask.reshape(*shape)
		mat = jfuncs.balance_margins(mat, k2, 0)
		mat = jfuncs.balance_margins(mat, k3, 1)
		return mat

	query_QW = _density_mask(qk, q_density, (Q, W))
	target_inds_W = jax.random.choice(tk, T, (W,)) 
	target_TW = jnp.full((T, W), False).at[target_inds_W,jnp.arange(W)].set(True)

	prod_TQW = query_QW[None,:,:] & target_TW[:,None,:]
	part_fn = jax.vmap(jax.vmap(jfuncs.partition_masked, in_axes=(None,0)), in_axes=(None,0))
	graph_inds_TQW, sizes_TQ = part_fn(jnp.arange(W), prod_TQW)
	graph_inds_TQW = jnp.where(
		jnp.arange(W)[None,None,:] < sizes_TQ[:,:,None], graph_inds_TQW, -1)
	trim = jnp.max(sizes_TQ)
	self.witness_inds_TQI = graph_inds_TQW[:,:,:trim]
	import pdb
	pdb.set_trace()
"""

