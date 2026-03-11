from dataclasses import asdict
from .generative import GenerativeModel, GenerativeModelOpts


__all__ = ['make_model']

def make_model(opts, seed: int):
	match opts:
		case GenerativeModelOpts():
			return GenerativeModel(opts, seed)
		case default:
			raise NotImplementedError

