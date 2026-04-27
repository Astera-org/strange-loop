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

	for step, item in enumerate(it):
		import pdb
		pdb.set_trace()
		tokens = np.array(item.obs_sym)
		for b in range(tokens.shape[0]):
			ds.validate(tokens[b])

if __name__ == "__main__":
	main()

