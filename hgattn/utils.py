import torch
import jax
import logging

def set_logger_level(logger_name: str, level: int):
	logger = logging.getLogger(logger_name)
	if logger is not None:
		# print(f"setting {logger_name} logger")
		logger.setLevel(level)

def quiet_loggers():
	for name in ("databricks.sdk", "jax._src.xla_bridge", "absl", "root"):
		set_logger_level(name, logging.WARNING)

def to_torch(ary: jax.Array) -> torch.Tensor:
	return torch.utils.dlpack.from_dlpack(ary)
	

