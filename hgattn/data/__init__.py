from typing import Any
from .types import TokensAndProbs

__all__ = ['TokensAndProbs', 'make_datasets', 'make_dataset']

def make_dataset(opts: Any, is_train: bool, seed: int) -> Any:
	from .copy_offset import CopyOffsetOpts, CopyOffsetDataset
	from .strided_count import StridedCountOpts, StridedCountDataset
	from .mod_addition import ModAdditionOpts, ModAdditionDataset
	from .expression import InductiveOpts, InductiveDataset
	from .polyseries import PolySeriesOpts, PolySeriesDataset
	match opts:
		case CopyOffsetOpts():
			return CopyOffsetDataset(opts)
		case StridedCountOpts():
			return StridedCountDataset(opts, is_train, seed)
		case ModAdditionOpts():
			return ModAdditionDataset(opts, is_train, seed)
		case InductiveOpts():
			return InductiveDataset(opts, is_train, seed)
		case PolySeriesOpts():
			return PolySeriesDataset(opts, is_train, seed) 
		case _:
			raise RuntimeError(f"Unrecognized dataset opts type: {type(opts)}")

		

