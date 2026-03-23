import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Any
from enum import Enum
# from .simple import SimpleCompModel
from ..layers import make_token_embed
from ..layers.embed import TokEmbedOpts
from ..layers.attn import AttentionOpts
from ..layers.block import TransformerBlock, NormType, FFNType
from .. import funcs
from ..data import TokensAndProbs
from .types import RunMode
from .. import rand
from .. import utils
from .. import logger
from ..debug import DebugOpts

class NormPattern(Enum):
	ALL = "all"
	SKIP_FIRST = "skip_first"

@dataclass
class GenerativeModelOpts:
	num_tokens: int
	model_dim: int
	hidden_dim: int
	num_heads: int
	d_head: int
	n_layers: int
	n_recurse: int
	norm_ty: NormType
	ffn_ty: FFNType
	norm_pat: NormPattern

	def __post_init__(self):
		try:
			self.norm_ty = NormType(self.norm_ty)
			self.ffn_ty = FFNType(self.ffn_ty)
			self.norm_pat = NormPattern(self.norm_pat)
		except Exception as ex:
			raise RuntimeError(f"One of norm_ty, ffn_ty, or norm_pat invalid") from ex


class GenerativeModel(nn.Module):
	"""
	A classic decoder-only autoregressive model.
	"""
	def __init__(
		self, 
		opts: GenerativeModelOpts, 
		attn_opts: AttentionOpts,
		tok_embed: TokEmbedOpts,
		dbg_opts: DebugOpts,
		seed: int
	): 
		super().__init__()
		self.opts = opts

		rng_state = torch.get_rng_state()
		torch.manual_seed(seed)

		self.embed = make_token_embed(tok_embed.ty, **tok_embed.args)
		self.final_norm = nn.RMSNorm(opts.model_dim)

		self.layers = nn.ModuleList()

		for i in range(opts.n_layers):
			match opts.norm_pat:
				case NormPattern.ALL:
					use_norm1 = use_norm2 = True
				case NormPattern.SKIP_FIRST:
					use_norm1 = (i == 0)
					use_norm2 = True
				case default:
					raise RuntimeError(f"Unrecognized NormPattern: {opts.norm_pat}")

			l = TransformerBlock(
				opts.model_dim, opts.num_heads, opts.d_head, attn_opts.qkv_bias,
				attn_opts.pos_ty, attn_opts.pos_args, opts.hidden_dim,
				opts.ffn_ty, opts.norm_ty,
				use_norm1, use_norm2
			)
			self.layers.append(l)

		self.unembed = nn.Linear(opts.model_dim, opts.num_tokens, bias=False)

		torch.set_rng_state(rng_state)
		self.log_probe_every = 10000

	@staticmethod
	def from_item(item: Any, train_targets_only: bool) -> dict:
		"""
		From a data item, return the arguments compatible with full 
		"""
		match item:
			case TokensAndProbs():
				label_mask = item.target_mask if train_targets_only else item.input_mask
				return dict(
					input_BC=item.obs_sym[:,:-1],
					input_mask_BC=item.input_mask[:,:-1],
					label_BC=item.obs_sym[:,1:],
					label_prob_BCV=item.obs_prob[:,1:],
					label_mask_BC=label_mask[:,1:],
					metric_mask_BC=item.target_mask[:,1:],
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

		x_BCM = self.embed(x_BC)

		for r in range(self.opts.n_recurse):
			for layer in self.layers:
				x_BCM = layer(x_BCM, full_mask_BQT)

		x_BCM = self.final_norm(x_BCM)
		out_BCV = self.unembed(x_BCM)
		return out_BCV

	def run(
		self,
		mode: RunMode,
		input_BC,       
		input_mask_BC,  # which input tokens are attended to
		label_BC,
		label_prob_BCV,
		label_mask_BC,  # which labels are used for gradients
		metric_mask_BC, # which predictions are used for the masked metrics
	) -> tuple[Tensor, Any]:
		match mode:
			case RunMode.TRAIN:
				pred_logit_BCV = self(input_BC, input_mask_BC)
			case RunMode.NOGRAD:
				pred_logit_BCV = funcs.run_no_grad(self, input_BC, input_mask_BC)
			case RunMode.MOCK:
				pred_logit_BCV = torch.log(label_prob_BCV + 1e-15)

		pred_logprob_BCV = torch.log_softmax(pred_logit_BCV, dim=2)

		xent_BC = funcs.cross_entropy(pred_logit_BCV, label_BC)
		xent = funcs.weighted_mean(xent_BC, label_mask_BC.to(xent_BC.dtype))

		kldiv_BC = funcs.kl_divergence(label_prob_BCV, pred_logprob_BCV).sum(axis=2)
		kldiv_masked = funcs.weighted_mean(kldiv_BC, metric_mask_BC.to(kldiv_BC.dtype))

		kldiv = kldiv_BC.mean()
		kldiv_label_mask = funcs.weighted_mean(kldiv_BC, label_mask_BC.to(kldiv_BC.dtype))

		acc = funcs.percent_correct(pred_logit_BCV, label_BC, label_mask_BC)
		acc_masked = funcs.percent_correct(pred_logit_BCV, label_BC, metric_mask_BC)

		return xent, { 
				"top1_acc": acc, 
				"top1_acc_mask": acc_masked,
				"kldiv": kldiv,
				"kldiv_mask": kldiv_masked,
				}

	def to_log_probe_data(self, step: int) -> list[dict[str, dict]]:
		if step % self.log_probe_every != 0: 
			return []

		results = []
		abbrev = dict(repeated_layers='l', attention='attn', embed='emb', disp_norm_HC='disp.hd')
		for path, buf in self.named_buffers():
			label = logger.map_probe_path(path, abbrev)
			if label is None:
				continue
			probe_data = logger.train_probe_data(step, label, buf)
			results.append(probe_data)
		return results

	def to_log_data(
		self,
		step: int,
		learning_rate: float,
		loss: Tensor,
		metrics: dict,
		data_split: str, # 'train' or 'test'
	) -> dict[str, dict]:

		data = { 
		  "training-3": 
			{ 
				 "xent": loss, 
				 "sgd_step": step,
				 "lr": learning_rate,
				 "data_split": data_split,
				 **metrics,
			} 
		}
		return data  

	def num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
