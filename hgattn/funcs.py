import torch
import torch.nn.functional as F

def masked_cross_entropy(
	pred_BCM: torch.Tensor,   # float[batch, context, model]
	targets_BC: torch.Tensor, # int[batch, context]
	mask_BC: torch.Tensor,    # bool[batch, context]
) -> torch.Tensor:

	pred_BMC = pred_BCM.permute(0,2,1)
	xent_BC = F.cross_entropy(pred_BMC, targets_BC)
	return (xent_BC * mask_BC.to(xent_BC.dtype)).mean()

def percent_correct(
	pred_BC: torch.Tensor,
	label_B: torch.Tensor, 
) -> float:
	correct_B = (pred_BC.argmax(axis=1) == label).to(torch.int32)
	return 100 * correct_B.sum() / correct_B.shape[0]

def max_is_correct(pred_C, label, mask) -> torch.Tensor:
	return torch.logical_and(pred_C.argmax() == label, mask)

def run_one_eval(model, *inputs):
	"""
	Convenience for running an eval of a model without affecting primals
	"""
	was_training = model.training
	model.eval()
	with torch.no_grad():
		output = model(*inputs)
	if was_training:
		# restore previous state
		model.train()
	return output

