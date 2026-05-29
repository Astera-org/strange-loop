from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Union, Any
import numpy as np
from enum import Enum


@dataclass
class StreamvisOpts:
    flush_every: float
    active: bool
    use_run_handle: str
    run_tags: list[str]

@dataclass
class TextLoggerOpts:
    path: str
    use_run_handle: str 
    run_tags: list[str]


def make_logger(opts: StreamvisOpts | TextLoggerOpts):
    match opts:
        case StreamvisOpts():
            from streamvis.logger import DataLogger
            return DataLogger(flush_every=opts.flush_every, dry_run=not opts.active)
        case TextLoggerOpts():
            return TextLogger(path=opts.path)


class TextLogger:
    def __init__(self, path: str):
        self.path = path
        try:
            self.fh = open(path, "w")
        except Exception as ex:
            raise RuntimeError(f"Couldn't open `{path}` for writing: {ex}")

    def start(self):
        pass

    def stop(self):
        pass

    def set_run_handle(self, handle: str):
        pass

    def set_run_attributes(self, /, **attrs):
        """Write a set of attributes to associate with this run.

        This is useful for recording hyperparameters, settings, configuration etc.
        for the program.  Can only be called once for the life of the logger.
        """
        pass

    def add_run_tags(self, *tags: list[str]):
        pass

    def write(self, series_name: str, /, **fields):
        """
        A `series_name` defines a set of named, typed fields (think C struct).
        `fields`.values() are assumed to be numpy|jax|pytorch arrays with
        broadcastable shapes. 
        """
        keys = tuple(fields.keys())
        values = []
        for k in keys:
            val = fields[k]
            match val:
                case int() | float() | str() | bool() | list() | tuple():
                    ary = np.array(val)
                case torch.Tensor():
                    ary = val.numpy(force=True)
                case jax.Array():
                    ary = npn.array(val)
                case _:
                    raise RuntimeError(f"Don't know how to convert a {type(val)} to numpy")
            values.append(ary)

        shapes = tuple(v.shape for v in values)
        full = np.broadcast_arrays(*values)
        flat = tuple(f.flatten() for f in full)
        for vals in zip(*flat):
            line = "\t".join(str(v) for v in vals)
            print(f"{series_name}\t{line}", file=self.fh)


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

