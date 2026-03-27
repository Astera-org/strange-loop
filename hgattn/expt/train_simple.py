"""
Run from the repo root:
    python -m hgattn.expt.train_simple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.graph_attn import GraphAttention_Naive
from ..layers.attn import PosEmbedType

# ── Config ────────────────────────────────────────────────────────────────────
CONTEXT_LEN      = 32
VOCAB_SIZE       = 16
TRAIN_VOCAB_SIZE = 12    # tokens during training drawn from 0..TRAIN_VOCAB_SIZE-1; full vocab at val
OP_FREQUENCY     = 0.1

D_MODEL        = 64
N_HEADS        = 1
N_LAYERS       = 1
FFN_HIDDEN_DIM = D_MODEL * 4

BATCH_SIZE     = 64
TRAIN_STEPS    = 12_000
LR             = 3e-4
WEIGHT_DECAY   = 0.1

if True: 
	print("Givens setup!!")
	USE_POS_INJECT   = True       # inject seq-position into dim D-1, scaled [-2, 2]
	USE_TOK_ENCODE   = True       # inject linear token id into dim D-2, scaled [-2, 2]
	USE_EMBED_MATRIX = False      # nn.Embedding for first D-2 dims; False → zeroed
	USE_RMS_NORM     = False      # True → RMSNorm (pre-LN); False → Identity
	USE_QK_NORM      = True       # post-proj Q,K RMSNorm inside attention
	POS_EMBED_TYPE   = PosEmbedType.GIVENS_RANDOM  # ROPE | GIVENS_RANDOM | GIVENS_ONE_HOT | NONE
else: 
	print("Rope setup!! (standard transformer)")
	USE_POS_INJECT   = False       # inject seq-position into dim D-1, scaled [-2, 2]
		# this is false with Rope b/c it internally calculates / represents position.  
	USE_TOK_ENCODE   = False       # inject linear token id into dim D-2, scaled [-2, 2]
		# don't need this either: just use the embedding matrix. 
	USE_EMBED_MATRIX = True       # nn.Embedding for first D-2 dims; False → zeroed
	USE_RMS_NORM     = True       # True → RMSNorm (pre-LN); False → Identity
	USE_QK_NORM      = False      # post-proj Q,K RMSNorm inside attention
	POS_EMBED_TYPE   = PosEmbedType.ROPE  # ROPE | GIVENS_RANDOM | GIVENS_ONE_HOT | NONE

USE_CE_LOSS      = USE_EMBED_MATRIX       # CE on first D-2 output dims → vocab logits
	# CE loss only makes sense if there is a one-hot embedding.
USE_MSE_LOSS     = USE_TOK_ENCODE  # MSE on dim D-2 of output vs linear token encoding
	# MSE loss only really makes sense if you linearly encode token value in the latent stream. 

REPORT_EVERY   = 200
# ─────────────────────────────────────────────────────────────────────────────


def generate_batch(
	batch_size: int,
	device: torch.device,
	vocab_size: int = TRAIN_VOCAB_SIZE,
) -> tuple[Tensor, Tensor, Tensor]:
	"""Generate a batch of CopyOffset data.

	At each position p, with probability OP_FREQUENCY the token value acts as a
	copy offset: the target is the token at position p - tokens[p].  Positions
	where that source index would be < 0 are silently dropped from the mask.

	Args:
	  vocab_size: token values drawn from 0..vocab_size-1.  Use TRAIN_VOCAB_SIZE
	              during training and VOCAB_SIZE for OOD validation.

	Returns:
	  tokens:       [B, C] int64  — random context
	  trigger_mask: [B, C] bool   — True at valid copy-offset positions (loss positions)
	  targets:      [B, C] int64  — source token to predict; meaningful only where trigger_mask
	"""
	B, C, V = batch_size, CONTEXT_LEN, vocab_size

	tokens = torch.randint(0, V, (B, C), device=device, dtype=torch.int64)

	# Candidate trigger positions: each True with probability OP_FREQUENCY
	trigger_mask = torch.rand(B, C, device=device) < OP_FREQUENCY

	# Source index: p - tokens[b, p]  (tokens value IS the offset)
	pos     = torch.arange(C, device=device)[None, :].expand(B, -1)  # [B, C]
	src_pos = pos - tokens                                           # [B, C], may be < 0

	# Invalidate triggers whose source would be out-of-bounds
	trigger_mask = trigger_mask & (src_pos >= 0)

	# Gather source tokens (clamp so gather doesn't error; invalid entries are masked out)
	targets = torch.gather(tokens, 1, src_pos.clamp(min=0))           # [B, C]

	return tokens, trigger_mask, targets


# ── Model ─────────────────────────────────────────────────────────────────────

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
		d_head             = d_model // n_heads
		self.d_model       = d_model
		self.embed         = TokenEmbed(vocab_size, d_model)
		norm_cls           = nn.RMSNorm if USE_RMS_NORM else nn.Identity
		self.blocks        = nn.ModuleList([
			TransformerBlock(d_model, n_heads, d_head, FFN_HIDDEN_DIM)
			for _ in range(n_layers)
		])
		self.norm_out      = norm_cls(d_model)
		# Unembedding reads dims 0..D-3 (excludes tok_enc and pos structural dims)
		self.unembed       = nn.Linear(d_model - 2, vocab_size, bias=False)
		# Learnable "copy trigger" vector added to input at trigger positions
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
		tokens: Tensor,        # [B, C] int64
		trigger_mask: Tensor,  # [B, C] bool  — loss positions (copy-offset argument tokens)
		targets: Tensor,       # [B, C] int64 — source token to predict at each trigger position
	) -> tuple[Tensor, dict]:
		B, C = tokens.shape
		device = tokens.device

		causal = torch.tril(
			torch.ones(C, C, dtype=torch.bool, device=device)
		).unsqueeze(0)                       # [1, C, C]

		out   = self.forward(tokens, causal, trigger_mask)  # [B, C, D]
		loss  = torch.tensor(0.0, device=device)
		n_tgt = trigger_mask.float().sum().clamp(min=1)
		metrics: dict[str, float] = {}

		if USE_CE_LOSS:
			logits  = self.unembed(out[:, :, :self.d_model - 2])    # [B, C, V]
			ce_all  = F.cross_entropy(
				logits.reshape(B * C, -1),
				targets.reshape(B * C),
				reduction='none',
			).reshape(B, C)
			ce_loss = (ce_all * trigger_mask.float()).sum() / n_tgt
			loss    = loss + ce_loss
			with torch.no_grad():
				acc = ((logits.argmax(-1) == targets) & trigger_mask).float().sum() / n_tgt
			metrics['ce_loss'] = ce_loss.item()
			metrics['acc']     = acc.item()

		if USE_MSE_LOSS:
			gt_enc   = targets.float() / (VOCAB_SIZE - 1) * 4.0 - 2.0  # [B, C]
			pred_enc = out[:, :, self.d_model - 2]                       # [B, C]
			mse_all  = (pred_enc - gt_enc).pow(2)
			mse_loss = (mse_all * trigger_mask.float()).sum() / n_tgt
			loss     = loss + mse_loss
			metrics['mse_loss'] = mse_loss.item()

		return loss, metrics

	def num_params(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

def sanity_check(device):
	# ── Sanity-check: print one example batch row ────────────────────────────
	with torch.no_grad():
		ex_tokens, ex_mask, ex_targets = generate_batch(4, device)
		print("\nExample batch (first 4 rows):")
		for b in range(4):
			tok_str = ' '.join(f"{t:2d}" for t in ex_tokens[b].tolist())
			msk_str = ' '.join((' *' if m else '  ') for m in ex_mask[b].tolist())
			tgt_str = ' '.join((f"{t:2d}" if m else ' .') for t, m in
			                   zip(ex_targets[b].tolist(), ex_mask[b].tolist()))
			print(f"  tokens:  {tok_str}")
			print(f"  trigger: {msk_str}")
			print(f"  targets: {tgt_str}")
			# Verify oracle: at each trigger position p, tokens[p-tokens[p]] == targets[p]
			C = ex_tokens.shape[1]
			for p in range(C):
				if ex_mask[b, p]:
					offset   = ex_tokens[b, p].item()
					src      = ex_tokens[b, p - offset].item()
					expected = ex_targets[b, p].item()
					ok = 'ok' if src == expected else f'MISMATCH(src={src})'
					print(f"    pos {p:2d}: offset={offset}, src@{p-offset}={src}, target={expected} {ok}")
			print()

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"device: {device}")

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

	# sanity_check(device)

	model.train()
	smoothing = 0.95
	ema: dict[str, float] = {}

	for step in range(TRAIN_STEPS):
		tokens, trigger_mask, targets = generate_batch(BATCH_SIZE, device, TRAIN_VOCAB_SIZE)

		optimizer.zero_grad()
		loss, metrics = model.compute_loss(tokens, trigger_mask, targets)
		loss.backward()
		optimizer.step()

		for k, v in metrics.items():
			ema[k] = smoothing * ema.get(k, v) + (1 - smoothing) * v

		if step % REPORT_EVERY == 0:
			m_str = '  '.join(f"{k}: {v:.4f}" for k, v in ema.items())
			print(f"step {step:6d}  loss: {loss.item():.4f}  {m_str}")

	print("done.")

	# ── OOD validation: full vocab ────────────────────────────────────────────
	VAL_BATCHES = 200
	model.eval()
	val_acc_sum  = 0.0
	val_mse_sum  = 0.0
	val_n        = 0
	with torch.no_grad():
		for _ in range(VAL_BATCHES):
			tokens, trigger_mask, targets = generate_batch(BATCH_SIZE, device, VOCAB_SIZE)
			_, metrics = model.compute_loss(tokens, trigger_mask, targets)
			if 'acc' in metrics:
				val_acc_sum += metrics['acc']
				val_n       += 1
			if 'mse_loss' in metrics:
				val_mse_sum += metrics['mse_loss']
	print(f"\nOOD validation ({VAL_BATCHES} batches, full vocab={VOCAB_SIZE}, "
	      f"train vocab={TRAIN_VOCAB_SIZE}):")
	if val_n:
		print(f"  acc:      {val_acc_sum / val_n:.4f}")
	if USE_MSE_LOSS:
		print(f"  mse_loss: {val_mse_sum / VAL_BATCHES:.4f}")


if __name__ == '__main__':
	main()
