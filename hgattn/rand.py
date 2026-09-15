import random
import torch
import os
import numpy as np
from contextlib import contextmanager

def get_system_random() -> int:
	raw = os.urandom(4)
	return int.from_bytes(raw, 'little')

def split_seed(seed: int, n: int) -> list[int]:
    seq = np.random.SeedSequence(seed)
    # Generate a 32-bit seed from each child sequence
    return [int(child.generate_state(1)[0]) for child in seq.spawn(n)]

def set_rand_state(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)


@contextmanager
def seeded_rng(seed: int):
	with torch.random.fork_rng():
		torch.manual_seed(seed)
		yield

