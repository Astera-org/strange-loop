import torch
import jax
import logging
import sys
from enum import EnumType, StrEnum

def quiet_loggers():
	for name in ("databricks.sdk", "jax._src.xla_bridge", "absl"):
		logger = logging.getLogger(name)
		if logger is not None:
			logger.setLevel(logging.WARNING)

def flush_loggers():
	for handler in logging.getLogger().handlers:
		handler.flush()

def to_torch(ary: jax.Array) -> torch.Tensor:
	return torch.utils.dlpack.from_dlpack(ary)
	
def make_log_exceptions_hook(logger):
	def log_exceptions(exc_type, exc_value, exc_traceback):
		if issubclass(exc_type, KeyboardInterrupt):
			sys.__excepthook__(exc_type, exc_value, exc_traceback)
			return
		logger.critical(
			"Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
		)
	return log_exceptions

class MirroredEnumMeta(EnumType):
	"""
	Metaclass that automatically adds member values as aliases.
	"""
	def __new__(metacls, name, bases, classdict, **kwds):
		extra_aliases = set()
		for key, val in list(classdict.items()):
			if not key.startswith("_") and isinstance(val, str):
				if val != key:
					extra_aliases.add(val)

		for alias in extra_aliases:
			classdict[alias] = alias

		return super().__new__(metacls, name, bases, classdict, **kwds)

class MirroredEnum(StrEnum, metaclass=MirroredEnumMeta):
	"""
	Use this for any StrEnums that are schema targets for OmegaConf conversion and
	checking, so that you can use the Enum value on the command-line

	Use as:

	class NormType(MirroredEnum):
	    RMS_NORM = "rmsnorm"
		LAYER_NORM = "layernorm"

	assert NormType.RMS_NORM == "rmsnorm"
	assert NormType.rmsnorm is NormType.RMS_NORM
	"""
	pass

