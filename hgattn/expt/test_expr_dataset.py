import sys
import os

if __package__ is None:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
	from hgattn import data
	from hgattn.data import iterator
	from hgattn import utils
	from hgattn import rand
else:
	from .. import data
	from ..data import iterator
	from .. import utils
	from .. import rand

import hydra
import numpy as np
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./opts", config_name="test_expr_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = instantiate(cfg)
	if opts.seed is None:
		opts.seed = rand.get_system_random()

	utils.quiet_loggers()	
	jnp.set_printoptions(threshold=sys.maxsize, floatmode="fixed", linewidth=200)

	ds = data.make_dataset(opts.data, opts.is_train, opts.seed)

	it = iterator.ShuffleIterator(
		dataset=ds, 
		num_elements=opts.dataset_size, 
		batch_size=opts.batch_size, 
		seed=opts.seed,
		new_epoch_cb=None,
		num_epochs=opts.num_epochs)

	if opts.do_mapreduce:
		def map_fn(item, *, bias):
			return item.obs_sym.sum() + bias
		def reduce_fn(accu, result):
			return accu + result
		mr = it.mapreduce(map_fn, reduce_fn, 0.0, {"bias": 3.0})
		print(f"mapreduce result:\n{mr}")

	for step, item in enumerate(it):
		tokens = np.array(item.obs_sym)
		active = np.array(item.active)

		for b in range(tokens.shape[0]):
			if not active[b]:
				continue
			print(ds.print(tokens[b]))
			# print(ds.print_raw(tokens[b]))
			pass

		if step % 100 == 0:
			print(f"step: {step}")
		for b in range(tokens.shape[0]):
			if not active[b]:
				continue
			success, msg = ds.validate(tokens[b])
			if not success:
				print(item.key[b], msg)

if __name__ == "__main__":
	main()

