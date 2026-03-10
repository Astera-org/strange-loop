import pathlib
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from ..opts import TrainOpts
from .. import data
from .. import models
from ..optim import OptimizerOpts, ScheduleOpts, build_schedule
from ..data.sampler import LoopedRandomSampler, ShuffleSampler, collate_pytree
from .. import funcs
from .. import sched
from ..layers.embed import PosEmbedType, TokEmbedType
from ..logger import Logger
from ..models.types import RunMode


@hydra.main(config_path="./opts", config_name="train_general", version_base="1.2")
def main(cfg: DictConfig):
	opts: TrainOpts = instantiate(cfg)
	train, test = data.make_datasets(opts.data)
	opts.arch.num_tokens = train.vocab_size
	match opts.arch.tok_embed.ty:
		case TokEmbedType.FIRST_N_MULT:
			opts.arch.tok_embed.args = {
				'ntoks': train.vocab_size, 
				'firstn': train.vocab_size - 1,
				'embed_dim': opts.arch.model_dim
			}
		case TokEmbedType.STANDARD:
			opts.arch.tok_embed.args = {
				'num_embeddings': train.vocab_size,
				'embedding_dim': opts.arch.model_dim,
			}

	logger = Logger(opts.logger) 
	if opts.logger.use_run_handle is not None:
		logger.set_run_handle(opts.logger.use_run_handle)

	logger.start()

	logger.set_run_attributes(
		tok_embed_type=opts.arch.tok_embed.ty.value
	)

	torch.set_printoptions(linewidth=210, threshold=1000000)

	train_loader = DataLoader(
			train, batch_size=opts.train.batch_size, 
			sampler=ShuffleSampler(len(train), opts.train.num_epochs),
			collate_fn=collate_pytree,
			pin_memory=True)

	test_loader = DataLoader(
			test, batch_size=opts.train.batch_size, pin_memory=True,
			sampler=LoopedRandomSampler(len(test)),
			collate_fn=collate_pytree,
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
	smoothing = 0.9
	ema_loss = torch.tensor(100.0, device=device)
	ema_train_acc = torch.tensor(0.0, device=device)

	def input_target(tensor):
		return tensor[:,:-1], tensor[:,1:] 

	test_iter = iter(test_loader)

	for item in train_loader:
		item = item.to(device)
		run_input = model.from_item(item)
		loss, metrics = model.run(RunMode.TRAIN, **run_input)
		ema_loss = funcs.update_ema(ema_loss, smoothing, loss.detach())

		mock_loss, mock_metrics = model.run(RunMode.MOCK, **run_input)

		sched.schedule_warmup_step(
			optimizer, opts.optim.learning_rate, opts.sched.warmup_steps, step
		)

		loss.backward()
		optimizer.step()

		lr = sched.get_optimizer_learning_rates(optimizer)[0]

		log_data = model.to_log_data(step, lr, loss, metrics, 'train')
		for series_name, field_data in log_data.items():
			logger.write(series_name, **field_data)

		m_log_data = model.to_log_data(step, lr, mock_loss, mock_metrics, 'mock')
		for series_name, field_data in m_log_data.items():
			logger.write(series_name, **field_data)

		if opts.train.do_test_metrics:
			t_item = next(test_iter)
			t_item = t_item.to(device)
			t_run_input = model.from_item(t_item)
			t_loss, t_metrics = model.run(RunMode.NOGRAD, **t_run_input)
			t_log_data = model.to_log_data(step, lr, t_loss, t_metrics, 'test')
			for series_name, field_data in t_log_data.items():
				logger.write(series_name, **field_data)

		if step % opts.train.report_every == 0:
			kldiv = metrics["kl_divergence"]
			mock_kldiv = mock_metrics["kl_divergence"]
			mock_acc = mock_metrics["percent_top_correct"]
			print(
					f"step: {step}, "
					f"train-loss: {loss.item():5.4f}, "
					f"kldiv: {kldiv.item():5.4f}, "
					f"mock-loss: {mock_loss.item():5.4f}, "
					f"mock-kldiv: {mock_kldiv.item():5.4f}, "
					f"mock-acc: {mock_acc.item():5.4f}"
					)

		if step % opts.sched.step_every == 0 and step > opts.sched.warmup_steps:
			scheduler.step(ema_loss)

		step += 1
	logger.stop()


if __name__ == "__main__":
	main()

