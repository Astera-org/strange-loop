import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Any
from .simple import SimpleCompModel
from ..layers.embed import EmbedType
from .. import funcs
from ..data import TokensAndProbs

@dataclass
class GenerativeModelOpts:
	num_tokens: int
	model_dim: int
	mlp_hidden_dim: int
	num_heads: int
	n_layers: int
	attn_impl: str
	pos_embed_type: EmbedType
	n_recurse: int

	def __post_init__(self):
		try:
			self.pos_embed_type = EmbedType(self.pos_embed_type)
		except ValueError as v:
			raise ValueError(
					f"Received invalid pos_embed_type `{self.ty.value}`.  "
					f"Valid ty's are {', '.join(m.value for m in EmbedType)}") from v


class GenerativeModel(SimpleCompModel):
	"""
	A classic decoder-only autoregressive model.
	"""

	def __init__(
			self, 
			num_tokens: int, 
			model_dim: int, 
			mlp_hidden_dim: int,
			num_heads: int,
			n_layers: int, 
			attn_impl: str='', 
			pos_embed_type: EmbedType=EmbedType.NONE,
			n_recurse: int=1
			): 
		super().__init__(num_tokens, model_dim, mlp_hidden_dim, num_heads, n_layers,
				   attn_impl, pos_embed_type, n_recurse)
		self.norm = nn.RMSNorm(model_dim)
		self.unembed = nn.Linear(model_dim, num_tokens, bias=False)

	@staticmethod
	def from_item(item: Any) -> dict:
		"""
		From a data item, return the arguments compatible with full 
		"""
		match item:
			case TokensAndProbs():
				return dict(
					input_BC=item.obs_sym[:,:-1],
					input_mask_BC=item.obs_mask[:,:-1],
					label_BC=item.obs_sym[:,1:],
					label_prob_BCV=item.obs_prob[:,1:],
					label_mask_BC=item.obs_mask[:,1:],
				)
			case default:
				raise NotImplementedError
	
	def forward(
			self,
			x_BC: Tensor,
			pad_mask_BT: Tensor,
			) -> Tensor:
		"""
		pad_mask_BC is False for PAD tokens, and will be combined with
		a causal mask to limit the self attention layers
		Input: 
			x_BC: int[batch, context]
			pad_mask_BT: bool[batch, target]
		Returns: 
			float[batch, context, token]
		"""
		C = x_BC.shape[1]
		causal_mask_QT = torch.arange(C)[:,None] >= torch.arange(C)[None,:]
		causal_mask_QT = causal_mask_QT.to(x_BC.device)
		full_mask_BQT = torch.logical_and(pad_mask_BT[:,None,:], causal_mask_QT[None,:,:])

		x_BCM = super().forward(x_BC, full_mask_BQT)
		x_BCM = self.norm(x_BCM)
		out_BCV = self.unembed(x_BCM)
		return out_BCV

	def run(
		self,
		input_BC,
		input_mask_BC,
		label_BC,
		label_prob_BCV,
		label_mask_BC,
	) -> tuple[Tensor, Any]:
		pred_logit_BCV = self.forward(input_BC, input_mask_BC)
		pred_logprob_BCV = torch.log_softmax(pred_logit_BCV, dim=2)
		xent = funcs.masked_cross_entropy(pred_logit_BCV, label_BC, label_mask_BC)
		kldiv_BC = funcs.kl_divergence(label_prob_BCV, pred_logprob_BCV).sum(axis=2)
		kldiv = funcs.weighted_mean(kldiv_BC, label_mask_BC.to(kldiv_BC.dtype))
		acc = funcs.percent_correct(pred_logit_BCV, label_BC, label_mask_BC)
		return xent, { "percent_top_correct": acc, "kl_divergence": kldiv }

	def to_log_data(
		self,
		step: int,
		learning_rate: float,
		loss: Tensor,
		metrics: dict,
		data_split: str, # 'train' or 'test'
	) -> dict[str, dict]:
		"""
		Format the output of 'run' to a dict with:
		series_name => field_data
		"""
		return { 
		  "training-1": 
			{ 
				 "cross_entropy": loss, 
				 "sgd_step": step,
				 "learning_rate": learning_rate,
				 "data_split": data_split,
				 **metrics,
			} 
		}




