import torch
from torch import nn
from .simple import SimpleCompModel

class GenerativeModel(SimpleCompModel):
	"""
	A classic decoder-only autoregressive model.
	"""

	def __init__(
		self, 
		num_tokens: int, 
		model_dim: int, 
        mlp_hidden_dim: int,
		output_dim: int, 
		num_heads: int,
		n_layers: int, 
		attn_impl: str='', 
		n_recurse: int=1
	): 
        super().__init__(num_tokens, model_dim, mlp_hidden_dim, num_heads, n_layers,
                         attn_impl, n_recurse)
		self.output_dim = output_dim
		self.unembed = nn.Linear(model_dim, num_tokens)

	def forward(
		self,
		x_BC: torch.Tensor,
		pad_mask_BT: torch.Tensor,
	) -> torch.Tensor:
		"""
		pad_mask_BT is False for PAD tokens, and will be combined with
		a causal mask to limit the self attention layers
		Input: 
			x_BC: int[batch, context]
			pad_mask_BT: bool[batch, target_pos]
		Returns: 
		    float[batch, context, token]
		"""
		C = x_BC.shape[1]
		causal_mask_QT = torch.arange(C)[:,None] >= torch.arange(C)[None,:]
		causal_mask_QT = causal_mask_QT.to(x_BC.device)
		full_mask_BQT = torch.logical_and(pad_mask_BT[:,None,:], causal_mask_QT[None,:,:])

		x_BCM = super().forward(x_BC, full_mask_BQT)
		out_BCV = self.unembed(x_BCM)
		return out_BCV


