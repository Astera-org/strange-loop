# Run from the repo root:
# python -m hgattn.expt.train_simple
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.graph_attn import GraphAttention_Naive
from ..layers.attn import PosEmbedType

CONTEXT_LEN      = 32
VOCAB_SIZE       = 16
TRAIN_VOCAB_SIZE = 10    # used only when VAL_MODE == 'max_offset': offsets 0..TRAIN_VOCAB_SIZE-1 in training
OP_FREQUENCY     = 0.1

# OOD generalisation mode — controls which offset values are seen during training:
#   'same'       — full vocab allowed as offsets in both train and val (no OOD gap)
#   'max_offset' — training offsets are 0..TRAIN_VOCAB_SIZE-1; val uses full vocab
# 			max_offset is useful for testing **extrapolation**
#   'random'     — a random 50% of offset values are withheld from training; val uses full vocab
# 			random is useful for testing **interpolation**
VAL_MODE = 'random'

D_MODEL        = 64
N_HEADS        = 1
N_LAYERS       = 1
FFN_HIDDEN_DIM = D_MODEL * 3

BATCH_SIZE     = 64
TRAIN_STEPS    = 12_000
LR             = 3e-4
WEIGHT_DECAY   = 0.1
USE_CAUSAL_MASK  = True # helps, but not neccesary.

# Givens config: explicit pos/tok injection + graph attention with one-hot Givens rotations
GIVENS_CONFIG = dict(
	USE_POS_INJECT   = True,   # inject seq-position into dim D-1, scaled [-2, 2]
	USE_TOK_ENCODE   = True,   # inject linear token id into dim D-2, scaled [-2, 2]
	USE_EMBED_MATRIX = True,   # nn.Embedding for first D-2 dims
	USE_RMS_NORM     = False,  # If True, RMS norm; o/w Identity.
		# speeds up learning, but prevents the network from learning a perfect solution. 
	USE_QK_NORM      = True,   # post-proj Q,K RMSNorm inside attention
		# not strictly needed, but seems to help? 
	POS_EMBED_TYPE   = PosEmbedType.GIVENS_RANDOM,  # ROPE | GIVENS_RANDOM | GIVENS_ONE_HOT | NONE
)

# RoPE config: standard transformer with rotary position embeddings
ROPE_CONFIG = dict(
	USE_POS_INJECT   = False,  # RoPE handles position internally
	USE_TOK_ENCODE   = False,  # embedding matrix is sufficient
	USE_EMBED_MATRIX = True,
	USE_RMS_NORM     = True,
	USE_QK_NORM      = False,
	POS_EMBED_TYPE   = PosEmbedType.ROPE,  # ROPE | GIVENS_RANDOM | GIVENS_ONE_HOT | NONE
)

globals().update(ROPE_CONFIG if '--rope' in sys.argv else GIVENS_CONFIG) 
	# nice trick from claude!

USE_CE_LOSS      = USE_EMBED_MATRIX
	# CE on first D-2 output dims → vocab logits
	# CE loss only makes sense if there is a one-hot embedding.
USE_MSE_LOSS     = USE_TOK_ENCODE
	# MSE on dim D-2 of output vs linear token encoding
	# MSE loss only really makes sense if you linearly encode token value in the latent stream. 

REPORT_EVERY   = 200

def make_offset_masks(device: torch.device) -> tuple[Tensor, Tensor]:
	"""Build (train_allowed, val_allowed) offset masks per VAL_MODE.

	Returns two [VOCAB_SIZE] bool tensors: True at offset values permitted in
	training / validation respectively.  val_allowed is always all-True.
	The random mask is sampled once here and shared so the split is consistent.
	"""
	val_allowed = torch.ones(VOCAB_SIZE, dtype=torch.bool, device=device)
	if VAL_MODE == 'same':
		return val_allowed.clone(), val_allowed
	if VAL_MODE == 'max_offset':
		train_allowed = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
		train_allowed[:TRAIN_VOCAB_SIZE] = True
		return train_allowed, val_allowed
	if VAL_MODE == 'random':
		perm = torch.randperm(VOCAB_SIZE, device=device)
		train_allowed = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
		train_allowed[perm[:VOCAB_SIZE // 2]] = True
		return train_allowed, val_allowed
	raise ValueError(f"Unknown VAL_MODE: {VAL_MODE!r}")

def generate_batch(
	batch_size: int,
	device: torch.device,
	allowed_offsets: Tensor,   # [VOCAB_SIZE] bool — which offset values are permitted
) -> tuple[Tensor, Tensor, Tensor]:
	"""Generate a batch of CopyOffset data.

	Tokens are always drawn from the full vocab (0..VOCAB_SIZE-1).  A trigger
	position is valid when its token value is in allowed_offsets and the source
	index p - tokens[p] is in bounds.  Candidate frequency is scaled up by
	VOCAB_SIZE / n_allowed so that ~OP_FREQUENCY triggers survive after filtering.

	Returns:
	  tokens:       [B, C] int64  — full-vocab random context
	  trigger_mask: [B, C] bool   — True at valid copy-offset positions (loss positions)
	  targets:      [B, C] int64  — source token to predict; meaningful only where trigger_mask
	"""
	B, C = batch_size, CONTEXT_LEN
	n_allowed = int(allowed_offsets.sum().item())

	tokens = torch.randint(0, VOCAB_SIZE, (B, C), device=device, dtype=torch.int64)

	adjusted_freq = min(OP_FREQUENCY * VOCAB_SIZE / max(n_allowed, 1), 1.0)
	trigger_mask  = torch.rand(B, C, device=device) < adjusted_freq

	# Source index: p - tokens[b, p]  (token value IS the offset)
	pos     = torch.arange(C, device=device)[None, :].expand(B, -1)  # [B, C]
	src_pos = pos - tokens                                             # [B, C]

	# Keep only triggers where offset is allowed and source is in bounds
	offset_ok    = allowed_offsets[tokens]                            # [B, C] bool
	trigger_mask = trigger_mask & offset_ok & (src_pos >= 0)

	targets = torch.gather(tokens, 1, src_pos.clamp(min=0))           # [B, C]
	return tokens, trigger_mask, targets


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
			qkv_bias=True, # NOTE!!  Helpful for Givens rotation! 
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
		tokens: Tensor,                           # [B, C] int64
		causal_mask: Tensor | None,               # [1, C, C] bool, or None for full attention
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

		if USE_CAUSAL_MASK:
			causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device)).unsqueeze(0)
		else:
			causal = None

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

def sanity_check(device, train_allowed: Tensor):
	train_set = sorted(i for i, ok in enumerate(train_allowed.tolist()) if ok)
	print(f"Training offsets ({len(train_set)}/{VOCAB_SIZE}): {train_set}")
	ex_tokens, ex_mask, ex_targets = generate_batch(4, device, train_allowed)
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
	cfg_name = 'rope' if '--rope' in sys.argv else 'givens'
	print(f"device: {device}  config: {cfg_name}")

	model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS).to(device)
	print(f"parameters: {model.num_params():,}")
	print(
		f"── data ──────────────────────────────────────────────────\n"
		f"  vocab_size={VOCAB_SIZE}  train_vocab_size={TRAIN_VOCAB_SIZE}  "
		f"context_len={CONTEXT_LEN}  op_freq={OP_FREQUENCY}\n"
		f"  val_mode={VAL_MODE!r}  "
		f"(train_vocab_size only used when val_mode='max_offset')\n"
		f"── optimiser ─────────────────────────────────────────────\n"
		f"  batch_size={BATCH_SIZE}  train_steps={TRAIN_STEPS}  "
		f"lr={LR}  weight_decay={WEIGHT_DECAY}\n"
		f"── model ─────────────────────────────────────────────────\n"
		f"  d_model={D_MODEL}  n_heads={N_HEADS}  n_layers={N_LAYERS}  "
		f"ffn_hidden={FFN_HIDDEN_DIM}\n"
		f"  pos_embed={POS_EMBED_TYPE.value}  pos_inject={USE_POS_INJECT}  "
		f"tok_encode={USE_TOK_ENCODE}\n"
		f"  embed_matrix={USE_EMBED_MATRIX}  rms_norm={USE_RMS_NORM}  "
		f"qk_norm={USE_QK_NORM}\n"
		f"  ce_loss={USE_CE_LOSS}  mse_loss={USE_MSE_LOSS}  causal_mask={USE_CAUSAL_MASK}\n"
		f"──────────────────────────────────────────────────────────"
	)

	optimizer = torch.optim.AdamW(
		model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
	)

	train_allowed, val_allowed = make_offset_masks(device)
	# sanity_check(device, train_allowed)

	model.train()
	smoothing = 0.95
	ema: dict[str, float] = {}

	for step in range(TRAIN_STEPS):
		tokens, trigger_mask, targets = generate_batch(BATCH_SIZE, device, train_allowed)

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
			tokens, trigger_mask, targets = generate_batch(BATCH_SIZE, device, val_allowed)
			_, metrics = model.compute_loss(tokens, trigger_mask, targets)
			if 'acc' in metrics:
				val_acc_sum += metrics['acc']
				val_n       += 1
			if 'mse_loss' in metrics:
				val_mse_sum += metrics['mse_loss']
	n_train = int(train_allowed.sum().item())
	print(f"\nOOD validation ({VAL_BATCHES} batches, mode={VAL_MODE!r}, "
	      f"train offsets={n_train}/{VOCAB_SIZE}):")
	if val_n:
		print(f"  acc:      {val_acc_sum / val_n:.4f}")
	if USE_MSE_LOSS:
		print(f"  mse_loss: {val_mse_sum / VAL_BATCHES:.4f}")


if __name__ == '__main__':
	main()
