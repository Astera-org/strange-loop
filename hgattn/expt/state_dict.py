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

	def __delitem__(self, key: int):
		val_V = self.state_KV[key,:]
		key_K = one_hot(key, self.max_keys)
		delta_KV = np.einsum('k, v -> kv', key_K, val_V)
		self.state_KV -= delta_KV

	def __setitem__(self, key: int, val: np.array):
		"""
		Update by adding the outer product of key vector and value vector
		"""
		key_K = one_hot(key, self.max_keys)
		delta_KV = np.einsum('k, v -> kv', key_K, val)
		self.state_KV += delta_KV
	
