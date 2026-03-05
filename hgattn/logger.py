from dataclasses import dataclass
from typing import Union, Any
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
	pass

class LoggerType(Enum):
	SV = 'streamvis'
	TXT = 'text'

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
				raise NotImplementedError
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
			case default:
				pass

	def flush_buffer(self):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.flush_buffer()
			case default:
				pass

	def set_run_handle(self, handle: str):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.set_run_handle(handle)
			case default:
				pass

	def set_run_attributes(self, attrs: dict):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.set_run_attributes(attrs)
			case default:
				pass

	def write(self, series_name: str, /, **field_values):
		match self.logger_type:
			case LoggerType.SV:
				return self._logger.write(series_name, **field_values)
			case default:
				pass

