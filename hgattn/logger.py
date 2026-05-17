from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Union, Any
import numpy as np
from enum import Enum
from streamvis.logger import DataLogger

@dataclass
class RunAttributes:
	args: dict[str, Any]

@dataclass
class StreamvisOpts:
    grpc_uri: str
    flush_every: float
    active: bool
    use_run_handle: str
    run_attrs: RunAttributes

def make_logger(opts: StreamvisOpts):
	logger = DataLogger(
		opts.grpc_uri,
		flush_every=opts.flush_every,
		dry_run=not opts.active,
	)
	return logger


def map_probe_path(
	path: str,
	abbrev: dict[str, str],
) -> str|None:
	legs = path.split(".")
	if not legs[-1].startswith("probe_"):
		return None

	out = []
	for leg in legs:
		leg = leg.replace("probe_", "")
		abbr = abbrev.get(leg, leg)
		out.append(abbr)
	return '.'.join(out)


def train_probe_data(
	sgd_step: int,
	path: str,
	buf: Tensor,
) -> dict[str, 'Array']:
	"""
	Format buffer data for the 'train-probe' series
	"""
	match buf.ndim:
		case 1: # [ctx_pos]
			ctx_pos = torch.arange(buf.numel())
			probe_loc = path
		case 2: # [dim2, ctx_pos]
			ctx_pos = torch.arange(buf.shape[1])[None,:]
			probe_loc = np.array([f"{path}.{i}" for i in range(buf.shape[0])])[:,None]
		case _:
			raise RuntimeError(f"buf must have 1 or 2 dimensions.  Got {buf.ndim=}")
	return { 
		 "train-probe":
		 {
			 "probe_loc": probe_loc,
			 "probe_val": buf,
			 "sgd_step": sgd_step,
			 "ctx_pos": ctx_pos,
		 }
	}

