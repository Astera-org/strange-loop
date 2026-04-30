import fire
from ..data import arith

def main(
	binops: list[str],
	uops: list[str],
	variables: list[str],
	consts: list[int],
	seed: int,
	max_depth: int,
	num_trees: int,
	global_mod_val: int=None  # will be applied to all mod_* binops and uops
):
	try:
		binops = tuple(arith.BinaryOp(b) for b in binops) 
		uops = tuple(arith.UnaryOp(u) for u in uops)
	except ValueError:
		raise RuntimeError(
			f"Valid binops are: {' '.join(b.value for b in arith.BinaryOp)}\n"
			f"Valid uops are: {' '.join(u.value for u in arith.UnaryOp)}\n"
		)

	tg = arith.TreeGen(binops, uops, variables, consts)
	trees = tg.gen_trees(seed, max_depth, num_trees)
	rpns = [arith.RPNExpression(tree, global_mod_val) for tree in trees]
	for rpn in rpns:
		print(rpn.infix())


if __name__ == "__main__":
	fire.Fire(main)

