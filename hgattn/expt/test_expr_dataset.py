import sys
import hydra
import numpy as np
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from .. import data
from ..data import iterator
from .. import utils
from .. import rand

@hydra.main(config_path="./opts", config_name="test_expr_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = instantiate(cfg)
	if opts.data_seed is None:
		opts.data_seed = rand.get_system_random()
	if opts.iter_seed is None:
		opts.iter_seed = rand.get_system_random()

	utils.quiet_loggers()	
	jnp.set_printoptions(threshold=sys.maxsize, floatmode="fixed", linewidth=200)

	ds = data.make_dataset(opts.data, opts.is_train, opts.data_seed)

	it = iterator.ShuffleIterator(
		dataset=ds, 
		num_elements=opts.dataset_size, 
		batch_size=opts.batch_size, 
		seed=opts.iter_seed,
		new_epoch_cb=None,
		num_epochs=opts.num_epochs)

	if opts.analyze_step is not None:
		item = it.get_batch_at_step(opts.analyze_step)
		passed, msg = ds.validate_item(item)
		if not passed:
			print(f"Item failed to validate:\n\n{msg}\n")
		ds.print_raw_item(item)
		import pdb
		pdb.set_trace()

	if opts.do_mapreduce:
		def map_fn(item, *, bias):
			return item.obs_sym.sum() + bias
		def reduce_fn(accu, result):
			return accu + result
		mr = it.mapreduce(map_fn, reduce_fn, 0.0, {"bias": 3.0})
		print(f"mapreduce result:\n{mr}")

	if opts.do_speedtest:
		print(f"starting speedtest")
		for item in it:
			# item = item.to_torch()
			if it.step_idx % 1000 == 0:
				print(f"step: {it.step_idx}")
		print(f"finished speedtest")

	if opts.do_validate:
		print("Validating...\n")
		for step, item in enumerate(it):
			if step % 100 == 0:
				print(f"step: {step}")
				passed, msg = ds.validate_item(item)
				if not passed:
					print(f"Item at step {step} failed to validate:\n\n{msg}")

	if opts.do_print_raw:
		for step, item in enumerate(it):
			if step % 100 == 0:
				print(f"step: {step}")
			print(ds.print_raw_item(item))

if __name__ == "__main__":
	main()

