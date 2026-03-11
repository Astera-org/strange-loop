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


@contextmanager
def seeded_rng(seed: int):
	with torch.random.fork_rng():
		torch.manual_seed(seed)
		yield

