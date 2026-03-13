import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ..opts import TestDatasetOpts
from ..data import iterator
from .. import data
from .. import utils
import jax.numpy as jnp
import jax


@hydra.main(config_path="./opts", config_name="test_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = instantiate(cfg)
	if opts.seed is None:
		opts.seed = rand.get_system_random()

	utils.quiet_loggers()	
	jnp.set_printoptions(threshold=sys.maxsize, floatmode="fixed", linewidth=200)

	ds = data.make_dataset(opts.data)
	print(OmegaConf.to_yaml(opts))

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
		tags = (item.key[:,0] % 10000).tolist()
		otags = list(sorted(tags))
		if step % 100 == 0:
			print(f"step: {step}, epoch: {it.epoch}, key_data: {tags}, key_data_sorted: {otags}")


if __name__ == "__main__":
	main()

