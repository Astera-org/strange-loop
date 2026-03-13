import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ..opts import TestDatasetOpts
from ..data import iterator
from .. import data
from .. import utils
import jax.numpy as jnp


@hydra.main(config_path="./opts", config_name="test_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = instantiate(cfg)
	if opts.seed is None:
		opts.seed = rand.get_system_random()

	utils.quiet_loggers()	
	jnp.set_printoptions(threshold=sys.maxsize, floatmode="fixed", linewidth=200)

	ds = data.make_dataset(opts.data)
	print(OmegaConf.to_yaml(opts.data))

	train_iter = iterator.ShuffleIterator(ds, 1000, opts.batch_size, opts.seed)

	import pdb
	for item in train_iter:
		pdb.set_trace()


if __name__ == "__main__":
	main()

