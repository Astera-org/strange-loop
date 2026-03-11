import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from ..opts import TestDatasetOpts
from .. import data


@hydra.main(config_path="./opts", config_name="test_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = instantiate(cfg)
	if opts.seed is None:
		opts.seed = rand.get_system_random()
	
	train, test = data.make_datasets(opts.data, opts.seed)

	for i in range(len(train)):
		item = train[i]
		import pdb
		pdb.set_trace()


if __name__ == "__main__":
	main()

