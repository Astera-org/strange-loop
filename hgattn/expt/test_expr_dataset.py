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
	opts: InductiveOpts = instantiate(cfg)
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

	import pdb
	pdb.set_trace()

	for step, item in enumerate(it):
		tokens = np.array(item.obs_sym)
		for b in range(tokens.shape[0]):
			# print(ds.print(tokens[b]))
			print(tokens)

		if step % 100 == 0:
			print(f"step: {step}")
		for b in range(tokens.shape[0]):
			success, msg = ds.validate(tokens[b])
			if not success:
				print(item.key[b], msg)

if __name__ == "__main__":
	main()

