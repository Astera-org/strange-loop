import torch
import torch.nn.functional as F

def masked_cross_entropy(
	pred_BCM: torch.Tensor,
	target_BCM: torch.Tensor,
	mask_BC: torch.Tensor,
	) -> torch.Tensor:

	target_BC = torch.argmax(target_BCM, axis=-1)
	xent_BC = F.cross_entropy(pred_BCM.permute(0,2,1), target_BCM.permute(0,2,1))
	with torch.no_grad():
		answers_BC = torch.argmax(pred_BCM, axis=2)
		n_correct = torch.where(mask_BC.to(torch.bool), answers_BC == target_BC, 0).sum()

	return {
			"loss": (xent_BC * mask_BC).sum(axis=1).mean(),
			"n-correct": n_correct,
			"n-possible": mask_BC.sum(),
			}


