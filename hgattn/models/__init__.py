from dataclasses import asdict
from .generative import GenerativeModel, GenerativeModelOpts


__all__ = ['make_model']

def make_model(opts):
	match opts:
		case GenerativeModelOpts():
			return GenerativeModel(opts)
		case default:
			raise NotImplementedError

