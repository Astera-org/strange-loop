import jax
from typing import Any
from torch.utils.data import Sampler
# from .melody import MelodyFactory, MelodyDataOpts
from .copy_offset import CopyOffsetOpts, CopyOffsetDataset
from .strided_count import StridedCountOpts, StridedCountDataset
from .mod_addition import ModAdditionOpts, ModAdditionDataset
from .expression import InductiveOpts, InductiveDataset
from .types import TokensAndProbs
from .. import rand 

from torch.utils.data import Dataset

__all__ = ['TokensAndProbs', 'make_datasets', 'make_dataset']

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
				opts.ctx_len, opts.use_cls_token, False,
				opts.num_tempos, opts.num_tempos_in_train,
				opts.max_melodies_to_use
			)
			return train, test
		case CopyOffsetOpts():
			train = CopyOffsetDataset(opts)
			test = CopyOffsetDataset(opts)
			return train, test
		case _:
			raise NotImplementedError

def make_dataset(opts: Any, is_train: bool, seed: int) -> Any:
	match opts:
		# case MelodyDataOpts():
		# 	raise NotImplementedError
		case CopyOffsetOpts():
			return CopyOffsetDataset(opts)
		case StridedCountOpts():
			return StridedCountDataset(opts, is_train, seed)
		case ModAdditionOpts():
			return ModAdditionDataset(opts, is_train, seed)
		case InductiveOpts():
			return InductiveDataset(opts, is_train, seed)
		case _:
			raise RuntimeError(f"Unrecognized dataset opts type: {type(opts)}")

		

