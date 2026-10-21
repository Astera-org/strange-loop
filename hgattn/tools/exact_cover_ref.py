import numpy as np
from numpy.random import Generator

def gen_mat(rng: Generator, R: int, C: int, avg_col_weight: float) -> np.ndarray:
	return rng.binomial(1, avg_col_weight, (R, C))

def exact_cover_naive(mat_RC: np.ndarray) -> tuple[np.ndarray, bool]:

	def _ec_rec(avail_R, avail_C, chosen_R) -> np.ndarray|None:
		if avail_C.sum() == 0:
			return chosen_R, True

		compat_RC = mat_RC * avail_R[:,None]

		# num_compat_C[c] = 0: dead column, > 0, < np.inf: in play, np.inf: NA
		num_compat_C = np.where(avail_C, compat_RC.sum(0), np.inf)

		if np.any(num_compat_C == 0):
			return chosen_R, False

		c = np.argmin(num_compat_C)
		cand_rows = np.flatnonzero(compat_RC[:,c])

		avail_C[c] = 0

		for r in cand_rows:
			clash_R = (mat_RC * mat_RC[r:r+1,:]).max(axis=1) 
			new_avail_R = avail_R * (1 - clash_R)
			chosen_R[r] = 1
			new_avail_C = avail_C * (1 - mat_RC[r,:])
			result = _ec_rec(new_avail_R, new_avail_C, chosen_R)
			if result is not None:
				return result, True
			chosen_R[r] = 0

		avail_C[c] = 1
		return chosen_R, False 

	R, C = mat_RC.shape
	return _ec_rec(np.ones(R), np.ones(C), np.zeros(R))

def exact_cover_loop(mat_RC: np.ndarray) -> tuple[np.ndarray, bool]:
	R, C = mat_RC.shape
	clash_RR = (np.einsum('rc, sc -> rs', mat_RC, mat_RC) > 0).astype(np.int32)
	block_R = np.zeros(R, np.int32) # block_R[r] = num rows blocking row r
	avail_C = np.ones(C, np.int32)
	chosen_R = np.zeros(R, np.int32)

	col_D = np.empty(C, np.int32)
	row_D = np.empty(C, np.int32)
	d, descend = 0, True

	while True:

		if descend:
			if avail_C.sum() == 0:
				return chosen_R, True
			compat_RC = mat_RC * (block_R[:,None] == 0)
			num_compat_C = np.where(avail_C, compat_RC.sum(0), np.inf)

			if np.any(num_compat_C == 0):
				descend = False
				continue

			c = np.argmin(num_compat_C)
			col_D[d] = c
			start = 0
		else:
			if d == 0:
				return chosen_R, False 
			d -= 1
			r = row_D[d]
			chosen_R[r] = 0
			block_R -= clash_RR[r]
			avail_C += mat_RC[r,:]
			c, start = col_D[d], r + 1

		compat_RC = mat_RC * (block_R[:,None] == 0)
		# cands = np.flatnonzero(compat_RC[start:,c])
		cands_R = np.where(np.arange(R) < start, 0, compat_RC[:,c])

		if cands_R.sum() == 0:
			# if len(cands) == 0:
			descend = False
			continue

		r = np.argmax(cands_R)
		# r = start + cands[0]
		row_D[d] = r
		block_R += clash_RR[r]
		avail_C -= mat_RC[r,:]
		chosen_R[r] = 1
		d += 1
		descend = True


def valid_exact_cover(
		mat_RC: np.ndarray, 
		chosen_R: np.ndarray,
		) -> bool:
	return np.all((mat_RC * chosen_R[:,None]).sum(axis=0) == 1)


def test(target: str, seed: int, R: int, C: int, avg_col_weight: float, reps: int) -> dict:
	match target:
		case "naive":
			exact_cover_fn = exact_cover_naive
		case "loop":
			exact_cover_fn = exact_cover_loop
		case _:
			raise RuntimeError(f"Unknown target: {target}")

	stats = dict(passed=0, failed=0, skipped=0)

	rng = np.random.default_rng(seed)

	for _ in range(reps):
		mat_RC = gen_mat(rng, R, C, avg_col_weight)
		chosen_R, found = exact_cover_fn(mat_RC)
		if found:
			if valid_exact_cover(mat_RC, chosen_R):
				stats['passed'] += 1
			else:
				stats['failed'] += 1
		else:
			stats['skipped'] += 1
	return stats

