"""
Tools for generating and manipulating Sudoku puzzles
"""
import numpy as np
from numpy.random import Generator

def solve_backtrack(board: np.ndarray) -> None:
	"""
	Solve the board in-place using backtracking
	"""
	pass


def is_valid_move(board: np.ndarray, move: int, loc: int) -> bool:
	if board[loc] != -1:
		return False
	r, c = divmod(loc, 9)
	br = r - (r % 3)
	bc = c - (c % 3)
	square = board.reshape(9, 9)
	return ( 
		 np.all(square[r,:] != move) 
		 and np.all(square[:,c] != move) 
		 and np.all(square[br:br+3,bc:bc+3] != move)
	)


def solvable(board: np.ndarray) -> bool:
	# Return whether board is solvable
    pass

def gen_full_board(rng: Generator) -> np.ndarray:
    # generate a full board 
	board = np.full((9, 9), -1)
	board[:3,:3] = rng.permutation(9).reshape(3, 3) + 1
	board[3:6,3:6] = rng.permutation(9).reshape(3, 3) + 1
	board[6:9,6:9] = rng.permutation(9).reshape(3, 3) + 1


	return board.flatten()


def generate(rng: Generator, nfull: int) -> np.ndarray:
    full = gen_full_board(rng)
	logits = np.full(81, 0.0)

    for i in reversed(range(nfull, 81)):
		u = rng.uniform(0.0, 1.0, 81)
		noise = -np.log(-np.log(u))
		while True:
			cell = np.argmax(logits + noise)
			prev_cell_value = board[cell]
			board[cell] = -1
			logits[cell] = -np.inf

			if solvable(board):
				break
			board[cell] = prev_cell_value 
			logits[cell] = 0.0
	return board

def make_puzzle(
	rng: Generator, 
	min_level: float, 
	max_level: float, 
	max_restarts=100
) -> np.ndarray:
	# make a puzzle within the difficulty level range
	board = 

