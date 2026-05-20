from hgattn.data import iterator
from hgattn.data.expression import InductiveOpts, InductiveDataset
from hgattn.tools.arith import BinaryOp, UnaryOp
import torch
import jax.numpy as jnp

seed=129384984
is_train=True

opts = InductiveOpts(
	int_base=None, # base encoding for large integers
	use_dpse=False, # use digit-position specific embeddings
	max_expr_depth=1,
	min_entropy_frac=0.9,
	allowed_num_consts=[1],
	allowed_num_vars=[1,2],
	n_exprs=678,
	binops=[BinaryOp.MOD_ADD, BinaryOp.MOD_SUB, BinaryOp.MOD_MUL],
	uops=[], # UnaryOp.MOD_SQR
	const_beg=0,
	const_end=113,
	input_beg=0,
	input_end=113,
	n_vars=1,
	n_consts=2,
	n_outputs=10,
	mod_val=113,
	train_frac=0.7,
	split_ty='input' # 'input' or 'expr'
)

# opts = InductiveOpts(
# 	int_base=None,
# 	use_dpse=False,
# 	max_expr_depth=3,
# 	min_entropy_frac=0.9,
# 	allowed_num_consts=[0,1],
# 	allowed_num_vars=[2,3],
# 	n_exprs=10000,
# 	binops=[BinaryOp.MOD_ADD, BinaryOp.MOD_SUB, BinaryOp.MOD_MUL],
# 	uops=[UnaryOp.MOD_SQR],
# 	const_beg=0,
# 	const_end=20,
# 	input_beg=0,
# 	input_end=113,
# 	n_vars=3,
# 	n_consts=10,
# 	n_outputs=10,
# 	mod_val=113,
# 	train_frac=0.7,
# 	split_ty='expr'
# )

dataset = InductiveDataset(opts, is_train, seed)

it = iterator.ShuffleIterator(
	dataset=dataset,
	num_elements=1000000,
	batch_size=5,
	seed=seed,
	new_epoch_cb=None,
	num_epochs=1000)

item = next(it)
torch_item = item.to_torch()

jnp.set_printoptions(linewidth=200, threshold=1000000)
torch.set_printoptions(linewidth=200, threshold=1000000)


print("Raw tokens:       ", item.obs_sym[0])
print("Human readable:   ", dataset.print(item.obs_sym[0]))
success, msg = dataset.validate(item.obs_sym[0])
print(f"Valid: {success}\nDetail:\n{msg}\n")
