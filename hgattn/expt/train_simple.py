"""

Run from the repo root (hgattn/..):
    python -m hgattn.expt.train_simple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.copy_offset import CopyOffsetDataset, CopyOffsetOpts
from ..data.iterator import ShuffleIterator
from ..layers.graph_attn import GraphAttention_Naive
from ..layers.attn import PosEmbedType
import pdb

# ── Config ────────────────────────────────────────────────────────────────────
CONTEXT_LEN    = 32
VOCAB_SIZE     = 16
OP_FREQUENCY   = 0.1

D_MODEL        = 64
N_HEADS        = 1
N_LAYERS       = 1
FFN_HIDDEN_DIM = D_MODEL * 4

BATCH_SIZE     = 64
TRAIN_STEPS    = 20_000
LR             = 3e-4
WEIGHT_DECAY   = 0.1

DATASET_SIZE   = 10_000   # virtual dataset size passed to ShuffleIterator
NUM_EPOCHS     = 100_000  # large enough that TRAIN_STEPS is the real limit
SEED           = 42

# Feature toggles
USE_POS_INJECT   = True           # inject seq-position into dim D-1, scaled [-2, 2]
USE_TOK_ENCODE   = True           # inject linear token id into dim D-2, scaled [-2, 2]
USE_EMBED_MATRIX = True           # nn.Embedding for first D-2 dims; False → zeroed
USE_RMS_NORM     = False          # True → RMSNorm (pre-LN); False → LayerNorm
USE_QK_NORM      = True          # post-proj Q,K RMSNorm inside attention
POS_EMBED_TYPE   = PosEmbedType.GIVENS_RANDOM     # ROPE | GIVENS_RANDOM | GIVENS_ONE_HOT | NONE
USE_CE_LOSS      = True           # CE on first D-2 output dims → vocab logits
USE_MSE_LOSS     = True           # MSE on dim D-2 of output vs linear token encoding

REPORT_EVERY   = 200
# ─────────────────────────────────────────────────────────────────────────────


class TokenEmbed(nn.Module):
	"""Embed integer tokens into D-dimensional vectors.

	Dimension layout (D = d_model):
	  dims  0 .. D-3 : nn.Embedding output        (if USE_EMBED_MATRIX, else zeros)
	  dim   D-2      : linear token encoding       (if USE_TOK_ENCODE, else zero)
	  dim   D-1      : position index              (if USE_POS_INJECT, else zero)
	CE unembedding reads dims 0..D-3; MSE reads dim D-2.
	"""

	def __init__(self, vocab_size: int, d_model: int):
		super().__init__()
		self.vocab_size = vocab_size
		self.d_model    = d_model
		if USE_EMBED_MATRIX:
			self.embedding = nn.Embedding(vocab_size, d_model - 2)

	def forward(self, tokens: Tensor) -> Tensor:
		# tokens: [B, C] int64
		B, C = tokens.shape

		if USE_EMBED_MATRIX:
			emb = self.embedding(tokens)                              # [B, C, D-2]
		else:
			emb = torch.zeros(B, C, self.d_model - 2,
			                  device=tokens.device, dtype=torch.float32)

		if USE_TOK_ENCODE:
			tok_enc = tokens.float() / (self.vocab_size - 1) * 4.0 - 2.0  # [B, C]
			tok_enc = tok_enc.unsqueeze(-1)                                  # [B, C, 1]
		else:
			tok_enc = torch.zeros(B, C, 1, device=tokens.device)

		if USE_POS_INJECT:
			pos = torch.arange(C, device=tokens.device, dtype=torch.float32)
			if C > 1:
				pos = pos / (C - 1) * 4.0 - 2.0
			pos = pos[None, :, None].expand(B, -1, 1)                # [B, C, 1]
		else:
			pos = torch.zeros(B, C, 1, device=tokens.device)

		return torch.cat([emb, tok_enc, pos], dim=-1)                # [B, C, D]


class TransformerBlock(nn.Module):
	def __init__(self, d_model: int, n_heads: int, d_head: int, hidden_dim: int):
		super().__init__()
		norm_cls   = nn.RMSNorm if USE_RMS_NORM else nn.Identity
		self.norm1 = norm_cls(d_model)
		self.norm2 = norm_cls(d_model)
		self.attn  = GraphAttention_Naive(
			d_model, n_heads, d_head,
			pos_embed_type=POS_EMBED_TYPE,
			pos_embed_args={},
			qkv_bias=True,
			qk_norm=USE_QK_NORM,
		)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, d_model),
		)

	def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
		x = x + self.attn(self.norm1(x), mask)
		x = x + self.ffn(self.norm2(x))
		return x


class SimpleTransformer(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		d_model: int,
		n_heads: int,
		n_layers: int,
	):
		super().__init__()
		d_head        = d_model // n_heads
		self.d_model  = d_model
		self.embed    = TokenEmbed(vocab_size, d_model)
		norm_cls      = nn.RMSNorm if USE_RMS_NORM else nn.Identity
		self.blocks   = nn.ModuleList([
			TransformerBlock(d_model, n_heads, d_head, FFN_HIDDEN_DIM)
			for _ in range(n_layers)
		])
		self.norm_out      = norm_cls(d_model)
		# Unembedding reads dims 0..D-3 (excludes tok_enc and pos structural dims)
		self.unembed       = nn.Linear(d_model - 2, vocab_size, bias=False)
		# Learnable "copy trigger" vector added to input 2 positions before each target
		self.trigger_embed = nn.Parameter(torch.zeros(d_model))

	def forward(
		self,
		tokens: Tensor,                      # [B, C] int64
		causal_mask: Tensor,                 # [1, C, C] bool
		trigger_mask: Tensor | None = None,  # [B, C] bool
	) -> Tensor:
		x = self.embed(tokens)               # [B, C, D]
		if trigger_mask is not None:
			x = x + trigger_mask.float().unsqueeze(-1) * self.trigger_embed
		for block in self.blocks:
			x = block(x, causal_mask)
		return self.norm_out(x)              # [B, C, D]

	def compute_loss(
		self,
		tokens: Tensor,       # [B, C] int64  (full context)
		target_mask: Tensor,  # [B, C] bool   (True = copy target position)
	) -> tuple[Tensor, dict]:
		B, C = tokens.shape
		device = tokens.device

		# Shift target_mask 2 positions left: the trigger fires at the offset-value token,
		# 2 steps before the actual copy target.
		# roll(..., -2): shifted[p] = target_mask[p+2]
		trigger_mask = torch.roll(target_mask, -2, dims=1).clone()
		trigger_mask[:, -2:] = False         # last 2 positions have no valid +2 target

		# Loss targets: the token 2 steps ahead of each trigger position
		# roll(tokens, -2): target_tokens[p] = tokens[p+2]
		target_tokens = torch.roll(tokens, -2, dims=1)  # [B, C]

		# Causal mask: token i attends only to positions 0..i
		causal = torch.tril(
			torch.ones(C, C, dtype=torch.bool, device=device)
		).unsqueeze(0)                       # [1, C, C]

		# Positive-control oracle: at each trigger position p, the token value IS the offset.
		# The source token lives at position p - offset = p - tokens[b,p].
		# Gather that token and compare to target_tokens to get perfect-algorithm accuracy.
		with torch.no_grad():
			pos      = torch.arange(C, device=device)[None, :].expand(B, -1)  # [B, C]
			src_pos  = (pos - tokens).clamp(min=0)                             # [B, C]
			oracle   = torch.gather(tokens, 1, src_pos)                        # [B, C]
			n_tgt_   = trigger_mask.float().sum().clamp(min=1)
			oracle_acc = ((oracle == target_tokens) & trigger_mask).float().sum() / n_tgt_

		# Full token sequence as input (no zeroing); trigger embedding added at trigger positions
		out = self.forward(tokens, causal, trigger_mask)  # [B, C, D]

		loss    = torch.tensor(0.0, device=device)
		metrics = {'oracle_acc': oracle_acc.item()}
		n_tgt   = trigger_mask.float().sum().clamp(min=1)

		if USE_CE_LOSS:
			logits  = self.unembed(out[:, :, :self.d_model - 2])    # [B, C, V]
			ce_all  = F.cross_entropy(
				logits.reshape(B * C, -1),
				target_tokens.reshape(B * C),
				reduction='none',
			).reshape(B, C)                                          # [B, C]
			ce_loss = (ce_all * trigger_mask.float()).sum() / n_tgt
			loss    = loss + ce_loss
			with torch.no_grad():
				acc = ((logits.argmax(-1) == target_tokens) & trigger_mask).float().sum() / n_tgt
			metrics['ce_loss'] = ce_loss.item()
			metrics['acc']     = acc.item()

		if USE_MSE_LOSS:
			# Ground truth: linear encoding of the token 2 steps ahead
			gt_enc   = target_tokens.float() / (VOCAB_SIZE - 1) * 4.0 - 2.0  # [B, C]
			pred_enc = out[:, :, self.d_model - 2]                             # [B, C]
			mse_all  = (pred_enc - gt_enc).pow(2)
			mse_loss = (mse_all * trigger_mask.float()).sum() / n_tgt
			loss     = loss + mse_loss
			metrics['mse_loss'] = mse_loss.item()

		return loss, metrics

	def num_params(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"device: {device}")

	# ── Data ─────────────────────────────────────────────────────────────────
	ds_opts  = CopyOffsetOpts(
		context_len=CONTEXT_LEN,
		vocab_size=VOCAB_SIZE,
		op_frequency=OP_FREQUENCY,
		fixed_offsets=None,
	)

	ds       = CopyOffsetDataset(ds_opts)
	train_it = ShuffleIterator(ds, DATASET_SIZE, BATCH_SIZE, SEED, None, NUM_EPOCHS)

	# ── Model ────────────────────────────────────────────────────────────────
	model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS).to(device)
	print(f"parameters: {model.num_params():,}")
	print(
		f"D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}  "
		f"pos_embed={POS_EMBED_TYPE.value}  pos_inject={USE_POS_INJECT}  "
		f"tok_encode={USE_TOK_ENCODE}  qk_norm={USE_QK_NORM}  "
		f"rms_norm={USE_RMS_NORM}  ce={USE_CE_LOSS}  mse={USE_MSE_LOSS}"
	)

	optimizer = torch.optim.AdamW(
		model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
	)

	# ── Training loop ────────────────────────────────────────────────────────
	model.train()
	smoothing = 0.95
	ema: dict[str, float] = {}

	for step, item in enumerate(train_it):
		# pdb.set_trace()
		if step >= TRAIN_STEPS:
			break

		item    = item.to_torch()
		tokens  = item.obs_sym.to(torch.int64).to(device)   # [B, C]
		t_mask  = item.target_mask.bool().to(device)         # [B, C]

		optimizer.zero_grad()
		loss, metrics = model.compute_loss(tokens, t_mask)
		loss.backward()
		optimizer.step()

		for k, v in metrics.items():
			ema[k] = smoothing * ema.get(k, v) + (1 - smoothing) * v

		if step % REPORT_EVERY == 0:
			m_str = '  '.join(f"{k}: {v:.4f}" for k, v in ema.items())
			print(f"step {step:6d}  loss: {loss.item():.4f}  {m_str}")

	print("done.")


if __name__ == '__main__':
	main()
