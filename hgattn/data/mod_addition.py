"""
A dataset of patterns: A B S A B S A B S ...
in which S = (A + B) % mod_val
where A and B are drawn from [0, vocab_size) and mod_val is in [1, vocab_size) 

The target positions are defined to be just the S role.

The train / test split is a random sampling of possible (A, B) pairs, with
train_split_frac of them given to the train_split.
"""


@dataclass
class ModAdditionOpts:
	context_len: int
	vocab_size: int
	mod_val: int
	train_split_frac: float  # fraction of possible (A, B) values given to train split

class ModAdditionDataset(eqx.Module):
	opts: ModAdditionOpts = eqx.field(static=True)

	def __init__(self, opts: ModAdditionOpts, is_train: bool, seed: int):
		self.opts = opts

	
	@property
	def vocab_size(self):
		return self.opts.vocab_size

	@eqx.filter_jit
	def _gen_item(self, key_B: Key) -> TokensAndProbs:

