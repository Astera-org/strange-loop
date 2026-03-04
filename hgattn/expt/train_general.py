import pathlib
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.func import vmap
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from ..opts import TrainOpts
from .. import data
from ..data import melody
from ..data.melody import MelodyDataOpts
from ..models.generative import GenerativeModel  
from ..models.simple import SimpleCompOpts
from ..optim import OptimizerOpts, ScheduleOpts, build_schedule
from ..data.sampler import LoopedRandomSampler, ShuffleSampler
from .. import funcs
from .. import sched
from ..layers.embed import EmbedType
from ..logger import Logger


@hydra.main(config_path="./opts", config_name="train_general", version_base="1.2")
def main(cfg: DictConfig):
	opts: TrainOpts = instantiate(cfg)
	train, test = data.make_datasets(opts.data)
	opts.arch.num_tokens = train.vocab_size

	logger = Logger(opts.logger) 
	logger.start()
	logger.set_run_handle()

	torch.set_printoptions(linewidth=210, threshold=1000000)

	train_loader = DataLoader(
			train, batch_size=opts.data.batch_size, 
			sampler=ShuffleSampler(len(train)),
			pin_memory=True)

	test_loader = DataLoader(
			test, batch_size=opts.data.batch_size, pin_memory=True,
			sampler=LoopedRandomSampler(len(test))
			)

	model = models.make_model(opts.arch)

	torch.set_float32_matmul_precision('high')

	if opts.train.do_compile:
		print("Compiling model")
		model = torch.compile(model)
		print("done.")

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print(f"Using device: {device}")

	model = model.to(device)
	num_params = model.num_params()
	print(f"Model has {num_params} parameters")
	print(f"Architecture:\n{opts.arch}\n")

	optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=opts.optim.learning_rate,
			betas=(opts.optim.b1, opts.optim.b2),
			eps=opts.optim.eps,
			weight_decay=opts.optim.weight_decay,
			)

	scheduler = build_schedule(optimizer, opts.sched)

	print("Start training")
	step = 0
	def make_map(device):
		def _mapfn(el):
			if isinstance(el, torch.Tensor):
				return el.to(device)
			return el
		return _mapfn

	map_fn = make_map(device)
	smoothing = 0.9
	ema_loss = torch.tensor(100.0, device=device)
	ema_train_acc = torch.tensor(0.0, device=device)

	def input_target(tensor):
		return tensor[:,:-1], tensor[:,1:] 

	test_iter = iter(test_loader)

	for item in train_loader:
		item = item.to(device)
		run_input = model.from_item(item)
		loss, metrics = model.run(**run_input)
		ema_loss = funcs.update_ema(ema_loss, smoothing, loss.detach())

		# pretty generic
		sched.schedule_warmup_step(
			optimizer, opts.optim.learning_rate, opts.warmup_steps, step
		)

		# generic
		loss.backward()
		optimizer.step()

		# custom metrics
		if opts.train.do_test_metrics:
			t_item = next(test_iter)
			t_item.to(device)
			t_run_input = model.from_item(t_item)
			t_loss, t_metrics = model.run(**t_run_input)

		# custom reports
		if step % opts.train.report_every == 0:
			lr = sched.get_optimizer_learning_rates(optimizer)[0]
			logmsg = (
					f"epoch: {epoch:3d}, "
					f"step: {step:6d}, "
					f"lr: {lr:20.15f}, "
					f"train-loss: {loss.item():5.4f} "
					f"ema-loss: {ema_loss.item():5.4f} "
					f"acc: {acc.item():5.4f} "
					)
			if opts.train.do_test_metrics:
				logmsg += (
						f"test-loss: {t_loss.item():5.4f} "
						f"test-acc: {t_acc.item():5.4f} "
						)
			print(logmsg)
			if arch.pos_embed_type == EmbedType.GIVENS:
				embed_norms = tuple(
						(l['attention'].embed.embed_weight ** 2).sum().item()
						for l in model.repeated_layers)
				pos_norms = tuple(
						(l['attention'].embed.pos_weight ** 2).sum().item()
						for l in model.repeated_layers)
				print(f"embed_norms: {embed_norms}")
				print(f"pos_norms: {pos_norms}")

		if step % opts.sched.step_every == 0 and step > opts.sched.warmup_steps:
			scheduler.step(ema_loss)

		step += 1
	logger.stop()


if __name__ == "__main__":
	main()

