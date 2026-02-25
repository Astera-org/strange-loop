import pathlib
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from hgattn.data import melody
from hgattn.data.melody import MelodyDataOpts
from hgattn.models.bertlike import BertlikeModel
from hgattn.models.simple import SimpleCompOpts
from hgattn.optim import OptimizerOpts, ScheduleOpts, build_schedule
from ..data.sampler import LoopedRandomSampler

@dataclass
class TempoInvariantOpts:
	arch: SimpleCompOpts
	data: MelodyDataOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	num_epochs: int
	report_every: int
	schedule_every: int
	do_compile: bool
	do_test_metrics: bool



@hydra.main(config_path="./opts", config_name="tempo_invariant", version_base="1.2")
def main(cfg: DictConfig):
	opts: TempoInvariantOpts = instantiate(cfg)
	fac = melody.MelodyFactory()
	path = pathlib.Path(opts.data.data_dir, opts.data.json_file)
	try:
		fac.load(path)
	except Exception as ex:
		raise RuntimeError(f"Couldn't load data from path: {path}")

	train, test = fac.get_datasets(
			opts.data.ctx_len, opts.data.use_cls_token, opts.data.output_onehot,
			opts.data.num_tempos, opts.data.num_tempos_in_train
			)
	train_loader = DataLoader(
			train, batch_size=opts.data.batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(
			test, batch_size=opts.data.batch_size, pin_memory=True,
			sampler=LoopedRandomSampler(len(test))
			)

	input_dim = fac.num_tokens
	output_dim = fac.num_classes

	arch = opts.arch
	model = BertlikeModel(
			input_dim, arch.hidden_dim, output_dim, arch.num_heads,
			arch.n_layers, arch.attn_impl, arch.n_recurse
			)

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

	def metrics(pred_BC, label):
		correct_B = (pred_BC.argmax(axis=1) == label).to(torch.int32)
		return 100 * correct_B.sum() / correct_B.shape[0]

	test_iter = iter(test_loader)

	def run_one_eval(model, *inputs):
		model.eval()
		with torch.no_grad():
			output = model(*inputs)
		model.train()
		return output

	for epoch in range(opts.num_epochs):
		for batch_idx, item in enumerate(train_loader):
			item = tree_map(map_fn, item)
			tokens, mask, label = (item[k] for k in ("notes-ids", "pad-mask", "output-class"))
			pred_BC = model(tokens, mask)
			train_acc = metrics(pred_BC, label)
			loss = F.cross_entropy(pred_BC, label) 

			ema_loss = smoothing * ema_loss + (1.0 - smoothing) * loss.detach() 
			ema_train_acc = smoothing * ema_train_acc + (1.0 - smoothing) * train_acc
			loss.backward()
			optimizer.step()
			step += 1
			loss = ema_loss

			if opts.do_test_metrics:
				test_item = next(test_iter)
				test_item = tree_map(map_fn, test_item)
				test_tokens, test_mask, test_label = (
				test_item[k] for k in ("notes-ids", "pad-mask", "output-class")
			)
				test_pred_BC = run_one_eval(model, test_tokens, test_mask)
				test_loss = F.cross_entropy(test_pred_BC, test_label)
				test_acc = metrics(test_pred_BC, test_label)

			if step % opts.report_every == 0:
				lr = scheduler.get_last_lr()[0]
				logmsg = (
						f"epoch: {epoch:3d}, "
						f"step: {step:6d}, "
						f"lr: {lr:20.15f}, "
						f"train-loss: {loss.item():5.4f} "
						f"train-acc: {train_acc.item():3.2f} "
						f"ema_loss: {ema_loss.item():5.4f} "
						f"ema_train_acc: {ema_train_acc.item():5.4f} "
						)
				if opts.do_test_metrics:
					logmsg += (
							f"test-loss: {test_loss.item():5.4f} "
							f"test-acc: {test_acc.item():5.4f} "
							)
				print(logmsg)

			if step % opts.schedule_every == 0:
				scheduler.step(ema_loss)






if __name__ == "__main__":
	main()
