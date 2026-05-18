import fire
import numpy as np

def one_hot(idx: int, num_classes: int):
	if idx >= num_classes:
		raise RuntimeError(f"idx must be less than num_classes")
	return np.eye(num_classes)[idx]

class OuterProductDict:
	"""
	A key-value store implemented as an outer product additive state update
	a la linear attention.

	Maps integer keys to vector values
	"""
	def __init__(self, max_keys: int, value_size: int):
		self.max_keys = max_keys
		self.state_KV = np.zeros((max_keys, value_size))
	
	def __getitem__(self, query: int) -> np.array:
		query_K = one_hot(query, self.max_keys)
		return np.einsum('kv, k -> v', self.state_KV, query_K)

	def __setitem__(self, key: int, val: np.array):
		"""
		Updates the value associated with key so that it's final value is `val`
		Implements the delta rule from 
		"""
		key_K = one_hot(key, self.max_keys)
		I_KK = np.eye(self.max_keys)
		H_KK = I_KK - np.einsum('a, b -> ab', key_K, key_K)
		new_mem_KV = np.einsum('k, v -> kv', key_K, val)
		self.state_KV = np.einsum('av, ab -> bv', self.state_KV, H_KK) + new_mem_KV
	
	def __delitem__(self, key: int):
		self[key] = np.zeros(self.value_size)
