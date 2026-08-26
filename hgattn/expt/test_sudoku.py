import fire
import numpy as np

from ..tools import sudoku


def main(seed: int, nfull: int):
	rng = np.random.default_rng(seed)
	board = generate(rng, nfull)
	print(board.tolist())

if __name__ == "__main__":
	fire.Fire(main)
