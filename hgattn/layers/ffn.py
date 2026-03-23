import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
	"""
	Swish Gated Linear Units based Feed-Forward Network.
	"""
	def __init__(self, in_features, hidden_features, out_features):
		super().__init__()
		self.w1 = nn.Linear(in_features, hidden_features, bias=False)
		self.w2 = nn.Linear(in_features, hidden_features, bias=False)
		self.w3 = nn.Linear(hidden_features, out_features, bias=False)

	def forward(self, x):
		return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MLP(nn.Module):
	"""
	Standard MLP layer
	"""
	def __init__(self, in_features, hidden_features, out_features):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_features, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, out_features)
		)

	def forward(self, x):
		return self.mlp(x)

