import pathlib
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from ..data import melody
from ..data.melody import MelodyDataOpts
from ..models.generative import GenerativeModel  
from ..models.simple import SimpleCompOpts
from ..optim import OptimizerOpts, ScheduleOpts, build_schedule
from ..data.sampler import LoopedRandomSampler
from .. import funcs

@dataclass
class SingSpeedOpts:
	arch: SimpleCompOpts
	data: MelodyDataOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	num_epochs: int
	report_every: int
	schedule_every: int
	do_compile: bool
	do_test_metrics: bool


@hydra.main(config_path="./opts", config_name="sing_speed", version_base="1.2")
def main(cfg: DictConfig):
	opts: SingSpeedOpts = instantiate(cfg)
	fac = melody.MelodyFactory()
	path = pathlib.Path(opts.data.data_dir, opts.data.json_file)
	try:
		fac.load(path)
	except Exception as ex:
		raise RuntimeError(f"Couldn't load data from path: {path}")

	train, test = fac.get_datasets(
			opts.data.ctx_len, opts.data.use_cls_token, False,
			opts.data.num_tempos, opts.data.num_tempos_in_train
			)
	train_loader = DataLoader(
			train, batch_size=opts.data.batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(
			test, batch_size=opts.data.batch_size, pin_memory=True,
			sampler=LoopedRandomSampler(len(test))
			)

	input_dim = fac.num_tokens
	output_dim = fac.num_tokens

	arch = opts.arch
	model = GenerativeModel(
			input_dim, arch.model_dim, arch.mlp_hidden_dim, output_dim,
			arch.num_heads, arch.n_layers, arch.attn_impl, arch.n_recurse
			)

	torch.set_float32_matmul_precision('high')

	if opts.do_compile:
		print("Compiling model")
		model = torch.compile(model)
		print("done.")

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print(f"Using device: {device}")

	model = model.to(device)

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

	for epoch in range(opts.num_epochs):
		for batch_idx, item in enumerate(train_loader):
			tokens, mask = (map_fn(item[k]) for k in ("notes-ids", "pad-mask"))
			input_BC, target_BC = input_target(tokens)
			input_mask_BC, target_mask_BC = input_target(mask)

			pred_BCM = model(input_BC, input_mask_BC)
			loss = funcs.masked_cross_entropy(pred_BCM, target_BC, target_mask_BC) 
			ema_loss = smoothing * ema_loss + (1.0 - smoothing) * loss.detach() 
			loss.backward()
			optimizer.step()

			if opts.do_test_metrics:
				test_item = next(test_iter)
				test_tokens, test_mask = (map_fn(test_item[k]) for k in ("notes-ids", "pad-mask"))
				t_input_BC, t_target_BC = input_target(t_tokens)
				t_input_mask_BC, t_target_mask_BC = input_target(t_mask)
				t_pred_BCM = funcs.run_one_eval(model, t_input_BC, t_mask_BC)
				t_loss = funcs.masked_cross_entropy(t_pred_BCM, t_target_BC, t_mask_BC) 

			if step % opts.report_every == 0:
				lr = scheduler.get_last_lr()[0]
				logmsg = (
						f"epoch: {epoch:3d}, "
						f"step: {step:6d}, "
						f"lr: {lr:20.15f}, "
						f"train-loss: {loss.item():5.4f} "
						f"ema-loss: {ema_loss.item():5.4f} "
						)
				if opts.do_test_metrics:
					logmsg += (
							f"test-loss: {t_loss.item():5.4f} "
							)
				print(logmsg)

			if step % opts.schedule_every == 0:
				scheduler.step(ema_loss)

			step += 1

if __name__ == "__main__":
	main()

