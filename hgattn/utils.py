import torch
import jax
import logging
import sys

def set_logger_level(logger_name: str, level: int):
	logger = logging.getLogger(logger_name)
	if logger is not None:
		# print(f"setting {logger_name} logger")
		logger.setLevel(level)

def quiet_loggers():
	for name in ("databricks.sdk", "jax._src.xla_bridge", "absl", "root"):
		set_logger_level(name, logging.WARNING)

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

	



