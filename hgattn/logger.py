from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Union, Any
import numpy as np
from enum import Enum

@dataclass
class RunAttributes:
	args: dict[str, Any]

@dataclass
class StreamvisOpts:
    active: bool
    grpc_uri: str
    flush_every: float
    use_run_handle: str
    run_attrs: RunAttributes

@dataclass
class TextLoggerOpts:
	path: str = "loss_log.txt"

class LoggerType(Enum):
	SV = 'streamvis'
	TXT = 'text'

def _val_to_str(v) -> str:
	if isinstance(v, torch.Tensor):
		if v.numel() == 1:
			return f"{v.item():.6g}"
		return str(v.tolist())
	if isinstance(v, float):
		return f"{v:.6g}"
	return str(v)

class Logger:
	def __init__(self, opts: Union[StreamvisOpts|TextLoggerOpts]):
		match opts:
			case StreamvisOpts():
				try:
					from streamvis.logger import DataLogger
				except ImportError as ie:
					raise RuntimeError(
						f"Requested a streamvis logger but could not import streamvis: {ie}")
				self._logger = DataLogger(
					grpc_uri=opts.grpc_uri,
					flush_every=opts.flush_every
				)
				self.logger_type = LoggerType.SV

			case TextLoggerOpts():
				self._file = open(opts.path, 'a')
				self.logger_type = LoggerType.TXT
			case default:
				raise RuntimeError(f"Unsupported opts type for logger: {type(opts)}")
	
	def start(self):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.start()
			case default:
				pass

	def stop(self):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.stop()
			case LoggerType.TXT:
				self._file.close()

	def set_run_handle(self, handle: str):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.set_run_handle(handle)
			case default:
				pass

	def set_run_attributes(self, /, **attrs):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.set_run_attributes(**attrs)
			case default:
				pass

	def write(self, series_name: str, /, **field_values):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.write(series_name, **field_values)
			case LoggerType.TXT:
				fields = "\t".join(f"{k}={_val_to_str(v)}" for k, v in field_values.items())
				self._file.write(f"{series_name}\t{fields}\n")
				self._file.flush()


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
		case default:
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

