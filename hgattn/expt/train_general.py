import random
import pathlib
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, asdict
import yaml
import torch
from torch.utils._pytree import tree_map
from ..opts import RunOpts
from ..optim import OptimizerOpts, ScheduleOpts, build_schedule
from ..data.iterator import ShuffleIterator
from ..layers.embed import TokEmbedType
from ..models.types import RunMode
from .. import funcs, sched, rand, utils, models
from ..data import make_dataset
from ..data.expression import TargetCategory # hack
from ..rand import get_system_random, split_seed
from ..logger import make_logger
from ..metrics import granular_metrics


def random31():
	return random.randint(0, 2**31 - 1)

OmegaConf.register_new_resolver("random_int", random31)

@hydra.main(config_path="./opts", config_name="train_general", version_base="1.2")
def main(cfg: DictConfig):
	utils.quiet_loggers()	

	opts: RunOpts = instantiate(cfg)

	if opts.seed is None:
		opts.seed = get_system_random()
	data_seed, model_seed = split_seed(opts.seed, 2)

	train = make_dataset(opts.data, True, data_seed)
	test = make_dataset(opts.data, False, data_seed)

	train_seed, test_seed = split_seed(data_seed, 2)

	train_batch_size = round(opts.train.batch_size * (opts.data.train_frac ** -1))
	test_batch_size = round(opts.train.batch_size * ((1 - opts.data.train_frac) ** -1))

	train_iter = ShuffleIterator(
		train, opts.train.train_dataset_size, train_batch_size,
		train_seed, None, opts.train.num_epochs)

	test_iter = ShuffleIterator(
		test, opts.train.test_dataset_size, test_batch_size,
		test_seed, None, opts.train.num_epochs)

	train_item = next(train_iter)
	context_len = train_item.obs_sym.shape[1]

	opts.arch.num_tokens = train.vocab_size
	if 'num_embeddings' in opts.embed.args:
		opts.embed.args['num_embeddings'] = train.vocab_size

	if 'ctx_len' in opts.embed.args:
		train_item = next(train_iter)
		opts.embed.args['ctx_len'] = context_len 

	loss_label_mask = 'copy_tokens_only' if opts.train.use_label_mask else 'all_tokens'

	logger = None
	# logger = make_logger(opts.logger)

	if logger and opts.logger.use_run_handle is not None:
		logger.set_run_handle(opts.logger.use_run_handle)

	# print(f"{train.vocab_size=} {train.num_digit_tokens=}")
	run_attrs = { k: v for k, v in opts.attrs.items() if v is not None }

	if logger:
		logger.start()

	logger.add_run_tags(*opts.logger.run_tags)

	logger.set_run_attributes(
		hparams=OmegaConf.to_yaml(cfg),
		**run_attrs,
		tok_embed_has_pos=opts.embed.args.get('splice_ctx_pos', False),
		trn_ctxlen=context_len,
		vocab_sz=train.vocab_size,
		loss_label_mask=loss_label_mask,
	)

	torch.set_printoptions(linewidth=210, threshold=1000000)

	model = models.make_model(opts.arch, opts.attn, opts.embed, opts.debug, model_seed)

	torch.set_float32_matmul_precision('high')

	# if opts.debug.do_compile:
	# 	print("Compiling model")
	# 	model = torch.compile(model)
	# 	print("done.")

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print(f"device: {device}")

	model = model.to(device)
	num_params = model.num_params()
	print(f"parameters: {num_params}")
	print(f"Architecture:\n{OmegaConf.to_yaml(opts.arch)}\n")
	print(f"Attention:\n{OmegaConf.to_yaml(opts.attn)}\n")
	print(f"Embed:\n{OmegaConf.to_yaml(opts.embed)}\n")
	print(f"Training:\n{OmegaConf.to_yaml(opts.train)}\n")
	print(f"Optim:\n{OmegaConf.to_yaml(opts.optim)}\n")
	print(f"LR Schedule:\n{OmegaConf.to_yaml(opts.sched)}\n")
	print(f"Data:\n{OmegaConf.to_yaml(opts.data)}\n")
	print(f"Attrs:\n{OmegaConf.to_yaml(opts.attrs)}\n")
	print(f"seed: {opts.seed}\n")

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
	ema_loss = None 
	last_ema_loss = torch.tensor(1000.0, device=device)

	train_iter.set_dataset_fraction(opts.train.start_ds_fraction)

	for item in train_iter:
		lr = sched.get_optimizer_learning_rates(optimizer)[0]

		item = item.to_torch()
		item.obs_sym = item.obs_sym.to(torch.int64)

		run_input = model.prepare_inputs(item, opts.train.use_label_mask)
		loss, metrics = model.run(
			RunMode.TRAIN, 
			run_input.input_BC,
			run_input.input_mask_BC,
			run_input.label_BC,
			run_input.label_prob_BCV,
			run_input.target_mask_BC)

		ema_loss = funcs.update_ema(ema_loss, smoothing, loss.detach())

		if logger:
			logger.write("metrics", sgd_step=step, lr=lr, xent=loss, data_split="train", **metrics)

		if opts.train.do_mock_metrics:
			mock_loss, mock_metrics = model.run(
				RunMode.MOCK, 
				run_input.input_BC,
				run_input.input_mask_BC,
				run_input.label_BC,
				run_input.label_prob_BCV,
				run_input.target_mask_BC)
			if logger:
				logger.write("metrics", sgd_step=step, lr=lr, xent=mock_loss, data_split="mock",
				**mock_metrics)
		else:
			mock_loss, mock_metrics = None, None

		sched.schedule_warmup_step(
			optimizer, opts.optim.learning_rate, opts.sched.warmup_steps, step
		)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# NOTE: deleted logging of model.to_log_probe_data here

		if opts.train.do_test_metrics:
			t_item = next(test_iter)
			t_item = t_item.to_torch()
			t_item.obs_sym = t_item.obs_sym.to(torch.int64)
			t_run_input = model.prepare_inputs(t_item, opts.train.use_label_mask)
			t_loss, t_metrics = model.run(
				RunMode.NOGRAD, 
				t_run_input.input_BC,
				t_run_input.input_mask_BC,
				t_run_input.label_BC,
				t_run_input.label_prob_BCV,
				t_run_input.target_mask_BC)
			if logger:
				logger.write(
				"metrics", sgd_step=step, lr=lr, xent=t_loss, data_split="test", **t_metrics)

		if opts.metric.active and abs(torch.log(ema_loss / last_ema_loss)) > opts.metric.step_interval:
			last_ema_loss = ema_loss

			for split in opts.metric.splits:
				match split:
					case 'train': metric_ds = train 
					case 'test': metric_ds = test
					case _: raise RuntimeError(f"Unknown split: {split}")

				gmetrics, counts, labels = granular_metrics(
					metric_ds, model, opts.metric.target_cats, opts.metric.num_samples,
					opts.metric.batch_size, opts.seed)

				gmetrics = tree_map(
					lambda d, sub: tree_map(lambda x: x / d, sub), 
					counts, gmetrics
				)
				# print(f"step: {step}, ema_loss: {ema_loss}, split: {split}")
				# print(gmetrics[TargetCategory.EXPR]["top1_acc"])

				if (ctx := gmetrics.get(TargetCategory.CTX_POS)) is not None:
					label = labels.get(TargetCategory.CTX_POS)
					if logger:
						logger.write(
							"metric-by-pos", sgd_step=step, data_split=split, 
							kldiv=ctx["kldiv"], top1_acc=ctx["top1_acc"], 
							ctx_pos=label)
				if (expr := gmetrics.get(TargetCategory.EXPR)) is not None:
					label = labels.get(TargetCategory.EXPR)
					if logger:
						logger.write(
							"metric-by-eqn", sgd_step=step, data_split=split,
							kldiv=expr["kldiv"], top1_acc=expr["top1_acc"],
							eqn_cat=label)

		if step % opts.debug.report_every == 0:
			m = metrics
			out = (
				f"step: {step}, "
				f"epoch: {train_iter.epoch}, "
				f"lr: {lr:10.8f}, "
				f"sampled-size: {train_iter.sampled_size}, "
				f"loss: {loss.item():5.4f}, "
				f"acc: {m['top1_acc'].item():5.4f}, "
				f"kldiv: {m['kldiv'].item():5.4f}, "
				)
			if opts.train.do_mock_metrics:
				mm = mock_metrics
				out += (
					f"mock-loss: {mock_loss.item():5.4f}, "
					f"mock-kldiv: {mm['kldiv'].item():5.4f}, "
					f"mock-acc: {mm['top1_acc'].item():5.4f}, "
					)
			print(out)

		if step % opts.sched.step_every == 0 and step > opts.sched.warmup_steps:
			scheduler.step(ema_loss)

		if step >= opts.train.max_sgd_steps:
			break

		step += 1
	if logger:
		logger.stop()


if __name__ == "__main__":
	main()

