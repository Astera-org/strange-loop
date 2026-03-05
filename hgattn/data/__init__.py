from typing import Any
from .arith import ExpressionDataset
from .melody import MelodyFactory, MelodyDataOpts 
from .copy_offset import CopyOffsetOpts, CopyOffsetDataset
from .types import TokensAndProbs

from torch.utils.data import Dataset

__all__ = ['TokensAndProbs', 'make_datasets']

def make_datasets(opts: Any) -> tuple[Dataset, Dataset]:
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
			train = CopyOffsetDataset(
				opts.context_len, opts.num_vals, opts.op_frequency,
				opts.dataset_size, opts.seed)
			test = CopyOffsetDataset(
				opts.context_len, opts.num_vals, opts.op_frequency,
				opts.dataset_size, opts.seed + opts.dataset_size)
			return train, test
		case default:
			raise NotImplementedError




