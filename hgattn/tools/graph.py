import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
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

def attention_weight(
	attn: Float[Array, "b i j k"],
	ctx_dim: Int[Array, "b c"],
	ctx_index: Int[Array, "b c"],
	dest_shape: tuple[int],
) -> Float[Array, "q w t"]:
	"""
	Tally attention scores according to the tokens they correspond to.
	Input:
	attn: attention scores over [i,j,k] cube of size [C,C,C]
	ctx_dim: a code in {0, 1, 2} representing which dimension of the output to target
	ctx_index: the index into the dimension to target
	dest_shape: the shape of the index
	
	Returns total attention weight accumulated by three token sub-vocabularies
	"""
	Q, W, T = dest_shape

	q = jnp.where(ctx_dim == 0, ctx_index, Q)
	w = jnp.where(ctx_dim == 1, ctx_index, W)
	t = jnp.where(ctx_dim == 2, ctx_index, T)
	out = jnp.zeros((Q, W, T), dtype=attn.dtype)
	return out.at[q,w,t].add(attn, mode="drop")

