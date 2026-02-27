from torch.optim.lr_scheduler import ReduceLROnPlateau



def schedule_warmup_step(
	optimizer,
	target_lr: float,
	num_warmup_steps: int,
	step: int
):
	if step >= num_warmup_steps:
		return
	current_lr = target_lr * (step / num_warmup_steps)
	for param_group in optimizer.param_groups:
		param_group['lr'] = current_lr

def get_optimizer_learning_rates(optimizer) -> list[float]:
	lrs = set(pg['lr'] for pg in optimizer.param_groups)
	return list(lrs)


