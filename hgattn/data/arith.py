import numpy as np
import torch
from torch.utils.data import Dataset
import math
import random

def graycodePosEnc(ntok, nbits, rand_phase=False):
	'''
	Generate a graycode
	seems more principled than standard SPE?
	'''
	pos_enc = np.zeros((ntok,nbits*2), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	if rand_phase:
		phase_offset = np.random.uniform() * 2 * math.pi
	else:
		phase_offset = 0
	for i in range(nbits):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		#   above is slower - does not help?
		period = 4 * 2.0**i
		if True:
			# sinusoidal, seems to work better?
			pos_enc[:, 2*i  ] = -np.cos(indx / period + phase_offset)
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset)
		else:
			# graycode! (thresholded)
			pos_enc[:, 2*i  ] = np.cos(indx / period + phase_offset) < 0
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset) < 0
	return pos_enc

def modOp(val1, val2, op, mod):
	match op:
		case 0:
			return (val1 + val2) % mod
		case 1:
			return (val1 - val2) % mod
		case 2:
			return (val1 * val2) % mod
		case 3:
			return (val1 // val2) % mod


"""
def modOp(va, vb, op, md):
	vc0 = (va + vb) % md
	vc1 = (va - vb) % md
	vc2 = (va * vb) % md
	if vb == 0:
		vc3 = 0 # can't divide by zero!
	else:
		vc3 = (va // vb) % md

	match op:
		case 0:
			vc = vc0
			ops = '+'
		case 1:
			vc = vc1
			ops = '-'
		case 2:
			vc = vc2
			ops = '*'
		case 3:
			vc = vc3
			ops = '/'
	return vc, ops
"""

OPERATORS = ['+', '-', '*', '/']

class Expression:

	def __init__(self, value=None, operator=None, left=None, right=None):
		self.value = value
		self.op = operator
		self.left = left
		self.right = right
		self.lparen_loc = 0
		self.loc = 0 # doubles for either value or op
		self.rparen_loc = 0

	def __str__(self):
		if self.op is None:
			return str(self.value)
		operator = OPERATORS[self.op]
		return f"({self.left} {operator} {self.right})"

	@classmethod
	def vocab(cls, md):
		return [*"()+-*/", *[str(v) for v in range(md)], cls.answer_token, cls.pad_token]

	def tokens(self):
		if self.op is None:
			return [str(self.value)]
		operator = OPERATORS[self.op]
		return ["(", *self.left.tokens(), operator, *self.right.tokens(), ")"]

	def tokenids(self, vocab_map) -> np.ndarray:
		"""
		Return a numpy array of token ids, assuming modulus md
		"""
		return np.array([vocab_map[v] for v in self.tokens()])

	def getLocs(self, width) -> np.ndarray:
		"""
		returns locs[pos, 3] where locs[pos,:] = (left, self, right) 
		"""
		n = self.count()
		if n > width:
			raise RuntimeError(f"Expression tokens exceed width: {n} >= {width}")

		locs = np.full((width, 2), -1, np.int32) # locs[l,:] = left, right

		def _rec(node):
			if node.op is None: # leaf node
				return
			locs[node.loc,0]= node.left.getLoc()
			locs[node.loc,1] = node.right.getLoc()
			_rec(node.left)
			_rec(node.right)

		_rec(self)
		return locs


	def setLocRec(self, loc):
		if self.op is None:
			self.loc = loc
			return loc + 1
		else:
			self.value = 0 # clear the value if it's an op
			self.lparen_loc = loc
			loc += 1
			loc = self.left.setLocRec(loc)
			self.loc = loc # operator
			loc += 1
			loc = self.right.setLocRec(loc)
			self.rparen_loc = loc
			loc += 1
			return loc

	def getLoc(self):
		return self.loc

	def printLoc(self):
		if self.value is not None:
			return str(self.loc)
		return f"({self.left.printLoc()} {self.loc} {self.right.printLoc()})"

	def count(self):
		if self.op is None:
			return 1 # ourself
		else:
			# 3 is for (,op,)
			return 3 + self.left.count() + self.right.count()

	def printParentLoc(self, parent):
		if self.op is None:
			return str(parent)
		return f"({self.left.printParentLoc(self.loc)} {parent} {self.right.printParentLoc(self.loc)})"


	def evaluate(self, md:int):
		# recusively evaluate the expression
		if self.op is None:
			return self.value
		self.value = c = modOp(self.left.evaluate(md), self.right.evaluate(md), self.op, md)
		return c

class ExpressionGenerator:
	"""Recursively generates random arithmetic expression trees."""

	def __init__(self, max_terms, modulo):
		self.max_terms = max(2, max_terms) # Need at least 2 terms for an op
		self.modulo = modulo
		# these catalan numbers start at 2.
		self.catalan = [2, 5, 14, 42, 132, 429, 1430, 4862]
		self.catalan[0] = 2 + 30 # increase the frequency of the
		self.catalan[1] = 5 + 20 # simple expr
		self.catalan[2] = 14 + 20 # our models r kiddos
		self.catalan_cumsum = np.cumsum(self.catalan)

	def generate(self):
		r = random.randrange(0, self.catalan_cumsum[self.max_terms-2])
		terms = np.sum(self.catalan_cumsum < r) + 2 # offset
		# print("terms:", terms)
		if terms > self.max_terms:
			pdb.set_trace()
		return self._generate_recursive(terms)

	def _generate_recursive(self, terms_count):
		"""The core recursive generation logic."""
		# Base case: if only one term is left, it must be a number.
		if terms_count <= 1:
			return Expression(value=random.randrange(self.modulo))

		op = random.randrange(4)

		# Split the remaining terms between left and right children.
		left_terms = random.randint(1, terms_count - 1)
		right_terms = terms_count - left_terms

		left_child = self._generate_recursive(left_terms)
		right_child = self._generate_recursive(right_terms)

		# Prevent division by the literal number 0.
		if op == 3:  
			while right_child.evaluate(self.modulo) == 0:
				right_child = self._generate_recursive(right_terms) # Reroll

		return Expression(operator=op, left=left_child, right=right_child)

class ExpressionDataset(Dataset):
	def __init__(
		self, 
		max_terms: int, 
		modulo: int, 
		dataset_size: int,
		context_len: int,
	):
		self.gen = ExpressionGenerator(max_terms, modulo)
		self.modulo = modulo
		self.context_len = context_len
		self.dataset_size = dataset_size
		self.tokens = [*"()+-*/", *[str(v) for v in range(modulo)], 'ANS', 'PAD']
		self.token_map = { v: i for i, v in enumerate(self.tokens) } 
		self.answer_token_id = self.tokens.index('ANS')
		self.pad_token_id = self.tokens.index('PAD')
		self.token_embed = torch.diag(torch.ones(len(self.tokens)))
		self.cache = [None] * self.dataset_size

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index: int):
		item = self.cache[index]
		if item is not None:
			return item

		val = 0
		while val == 0:
			tree = self.gen.generate()
			val = tree.evaluate(self.modulo)
		toks = tree.tokenids(self.token_map)
		answer_tok_pos = toks.shape[0] # answer is right after last expression token
		input_toks = np.full(self.context_len, self.pad_token_id)
		input_toks[:answer_tok_pos] = toks
		input_toks[answer_tok_pos] = self.answer_token_id
		input_toks[answer_tok_pos + 1] =  self.pad_token_id
		inputs = self.token_embed[input_toks]

		target_toks = np.full(self.context_len, self.pad_token_id)
		target_toks[answer_tok_pos + 1] = self.token_map[str(val)]
		targets = self.token_embed[target_toks]

		loss_mask = torch.Tensor(target_toks != self.pad_token_id).to(dtype=torch.int32)

		item = { 
		  "input": inputs[:-1,:],
		  "target": targets[1:,:],
		  "loss_mask": loss_mask[1:],
		}
		self.cache[index] = item
		return item
		

