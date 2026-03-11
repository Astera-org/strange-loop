from typing import Any
from .arith import ExpressionDataset
from .melody import MelodyFactory, MelodyDataOpts 
from .copy_offset import CopyOffsetOpts, CopyOffsetDataset
from .types import TokensAndProbs
from .. import rand 

from torch.utils.data import Dataset

__all__ = ['TokensAndProbs', 'make_datasets']

def make_datasets(opts: Any, seed: int) -> tuple[Dataset, Dataset]:
	match opts:
		case MelodyDataOpts():
			fac = MelodyFactory()
			path = pathlib.Path(opts.data_dir, opts.json_file)
			try:
				fac.load(path)
			except Exception as ex:
				raise RuntimeError(f"Couldn't load melody data from path: {path}")
			train, test = fac.get_datasets(
				opts.ctx_len, opts.use_cls_tokken, False,
				opts.num_tempos, opts.num_tempos_in_train,
				opts.max_melodies_to_use
			)
			return train, test
		case CopyOffsetOpts():
			train_seed, test_seed = rand.split_seed(seed, 2)
			train = CopyOffsetDataset(opts, train_seed)
			test = CopyOffsetDataset(opts, test_seed)
			return train, test
		case default:
			raise NotImplementedError


