import torch
import torch.nn.functional as F
from torch import nn
from torch.func import vmap
from torch import Tensor


def masked_cross_entropy(
	pred_logit_BCM: torch.Tensor,   # float[batch, context, model]
	targets_BC: torch.Tensor, # int[batch, context]
	mask_BC: torch.Tensor,    # bool[batch, context]
) -> torch.Tensor:

	pred_logit_BMC = pred_logit_BCM.permute(0,2,1)
	xent_BC = F.cross_entropy(pred_logit_BMC, targets_BC)
	return weighted_mean(xent_BC, mask_BC.to(xent_BC.dtype))

def kl_divergence(p: Tensor, qlog: Tensor) -> Tensor:
	assert p.shape == qlog.shape, "p and qlog must have identical shapes"
	term = p * (torch.log(p) - qlog)
	return torch.where(p == 0, 0.0, term)

def weighted_mean(values, weights):
	return (values * weights).sum() / weights.sum()

def update_ema(
	ema_values: torch.Tensor, 
	smoothing: float,
	values: torch.Tensor
) -> torch.Tensor:
	return smoothing * ema_values + (1.0 - smoothing) * values


def percent_correct(
	pred_BC: torch.Tensor,
	label_B: torch.Tensor, 
) -> float:
	correct_B = (pred_BC.argmax(axis=1) == label).to(torch.int32)
	return 100 * correct_B.sum() / correct_B.shape[0]

def max_is_correct(pred_C, label, mask) -> torch.Tensor:
	return torch.logical_and(pred_C.argmax() == label, mask)

def run_no_grad(model: nn.Module, *args, **kwargs):
	"""
	Convenience for running an eval of a model without affecting primals
	"""
	was_training = model.training
	model.eval()
	with torch.no_grad():
		output = model(*args, **kwargs)
	if was_training:
		# restore previous state
		model.train()
	return output


def percent_correct(pred_BCM, target_BC, active_BC):
	"""
	Computes the percent that the maximum prediction (over axis 2)
	coincides with the target category, only considering the [batch, context]
	positions marked active.
	"""
	correct_fn = vmap(vmap(max_is_correct))
	relevant_BC = correct_fn(pred_BCM, target_BC, active_BC)
	n_correct = relevant_BC.to(torch.int32).sum()
	n_total = active_BC.to(torch.int32).sum()
	return (n_correct / n_total) * 100.0


def find_first_value(target: Tensor, value: int|float) -> Tensor:
	"""
	Given a tensor of target values, and a test value,
	return a 0D tensor with either the index such that target[index] == value
	or -1 if it doesn't exist
	"""
	mask = (target == value)
	exists = mask.any()
	index = mask.float().argmax()
	return torch.where(
		exists, index.to(torch.long), torch.tensor(-1, device=target.device)
	)






