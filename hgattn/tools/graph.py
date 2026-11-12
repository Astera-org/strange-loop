import numpy as np
from numpy.random import Generator

def gen_witness_graph(
    R: int, W: int, C: int, n_witness_per_link: int, rng: Generator
) -> np.array:
	graph = np.zeros((R, W, C), dtype=np.int32)
	rand_vals = rng.random((R, W, C))
	topk_RWC = np.argpartition(rand_vals, n_witness_per_link, axis=1)
	k_inds = topk_RWC[:,:n_witness_per_link,:]
	graph[np.arange(R)[:,None,None], k_inds, np.arange(C)[None,None,:]] = 1
	return graph

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


