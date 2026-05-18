import torch
from torch import nn

class UniformAttention(nn.Module):
	"""
	A mock version of attention that is not trainable and simply combines context
	uniformlyl

	"""
	def __init__(self):
		super().__init__()

	def _forward(self, x_CM, target_mask):
		"""
		x_CM: [context, model]
		target_mask: one of:
			bool[target] - a target mask applied to all queries
			bool[query, target] - a target mask specific to each query
		"""
		C, M = x_CM.shape
		A_QT = torch.ones((C,C), device=x_CM.device)

		if target_mask.ndim == 1: # [target]
			target_mask = target_mask[None,:]
		A_QT = torch.where(target_mask, A_QT, torch.full_like(A_QT, float('-inf')))
		A_QT = torch.softmax(A_QT, dim=-1)
		y_QM = torch.einsum('qt, tm -> qm', A_QT, x_CM)
		return y_QM

	def forward(self, x_BCM, target_mask):
		return torch.vmap(self._forward, in_dims=(0, 0))(x_BCM, target_mask)





