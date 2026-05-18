import random
import fire
from ..tools import arith

def main(
	binops: list[str],
	uops: list[str],
	variables: list[str],
	consts: list[int],
	seed: int,
	max_depth: int,
	num_trees: int,
	allowed_var_nums: list[int], # list of allowed numbers of variables in the formulas
	global_mod_val: int,  # will be applied to all mod_* binops and uops
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
	rpns = [rpn for rpn in rpns if len(rpn.variables) in allowed_var_nums]

	tokens = binops + uops + tuple(arith.ControlOp) + tuple(variables)
	op_map = { k: i for i, k in enumerate(tokens) }
	zero_token = max(op_map.values()) + 1

	for rpn in rpns:
		kwargs = { v: random.randint(0, global_mod_val) for v in rpn.variables }
		kwarg_string = ", ".join(f"{k}={v}" for k, v in kwargs.items())
		rpn_enc = rpn.tokens_base_enc(op_map, use_dpse=True, base=10, zero=zero_token)
		print(
				f"expression: {rpn.infix()}\n"
				f"variables: {", ".join(rpn.variables)}\n"
				f"constants: {", ".join(str(c) for c in rpn.const_values)}\n"
				f"expr({kwarg_string}): {rpn.evaluate(**kwargs)}\n"
				f"tokens: {" ".join(str(t) for t in rpn_enc)}\n\n")

	

if __name__ == "__main__":
	fire.Fire(main)

