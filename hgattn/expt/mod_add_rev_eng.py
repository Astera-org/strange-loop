"""
Reverse engineered network for modular addition
"""
import fire
import math
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
import optax
from ..logger import make_logger, StreamvisOpts

def make_svg_grid(
		ref_RC2: jax.Array,
		test_RCM: jax.Array,
		margin: int=50,  # space around the main grid
		arrow_frac: float=0.3,  # fraction of the cell the arrow takes up
		stroke: float=2.0,
		cell: int=50,
		) -> str:
	"""

	"""
	test_RCM = np.array(test_RCM)
	M = test_RCM.shape[2]
	tests_RC2 = tuple(test_RCM[:,:,off:off+2] for off in range(0, M, 2))
	ref_RC2 = np.array(ref_RC2)
	nr, nc, _ = ref_RC2.shape

	height = nr * cell + 2 * margin
	width = nc * cell + 2 * margin
	L = arrow_frac * cell          # on-screen length of a "unit" arrow
	half = L / 2.0

	# Arrowhead geometry (drawn via a reusable marker).
	head_len = 0.25 * L
	head_w = 0.25 * L

	out = []
	out.append(
			f'<svg xmlns="http://www.w3.org/2000/svg" '
			f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
			)
	out.append(f'  <rect width="{width}" height="{height}" fill="white"/>')

	# One marker, auto-oriented along each line, sized in user units.
	out.append('  <defs>')
	out.append(
			f'    <marker id="head" markerUnits="userSpaceOnUse" '
			f'markerWidth="{head_len}" markerHeight="{head_w}" '
			f'refX="0" refY="{head_w / 2}" orient="auto">'
			)
	out.append(
			f'      <path d="M0,0 L{head_len},{head_w / 2} L0,{head_w} Z" '
			f'fill="context-stroke"/>'
			)
	out.append('    </marker>')
	out.append('  </defs>')

	out.append(
			f'  <g stroke="black" stroke-width="{stroke}" '
			f'fill="none" marker-end="url(#head)">'
			)

	def draw_one(i, j, data_RC2, color):
		tail_x = margin + (j + 0.5) * cell
		tail_y = margin + (i + 0.5) * cell
		dx = data_RC2[i,j,0] * L
		dy = data_RC2[i,j,1] * L
		seg = (dx * dx + dy * dy) ** 0.5
		if seg < 1e-9:
			return
		ux, uy = dx / seg, dy / seg
		tip_x, tip_y = tail_x + dx, tail_y - dy
		neck_x, neck_y = tip_x - head_len * ux, tip_y - head_len * uy

		# SVG's y-axis points downward, so negate dy to keep the
		# angle counter-clockwise from the positive x-axis on screen.
		out.append(
				f'    <line x1="{tail_x:.3f}" y1="{tail_y:.3f}" '
				f'x2="{neck_x:.3f}" y2="{neck_y:.3f}" stroke="{color}"/>'
				)

	colors = "red", "orange", "yellow", "green", "blue", "purple", "brown"

	for i in range(nr):
		for j in range(nc):
			for k, test_RC2 in enumerate(tests_RC2):
				draw_one(i, j, test_RC2, colors[k])
			draw_one(i, j, ref_RC2, "black")

	out.append('  </g>')
	out.append('</svg>')
	return '\n'.join(out)

def wheels(freqs: jax.Array, phases: jax.Array, window_size: int) -> jax.Array:
	"""
	Produce a fourier series with frequencies freqs * 2 \\pi / window_size
	freqs: number of full revolutions per `window_size` steps
	Returns f4[window_size, 2F]
	"""
	F = freqs.shape[0]
	rad_per_win = 2.0 * jnp.pi / window_size
	index_VF = jnp.arange(window_size, dtype=jnp.float32)[:,None] + phases[None,:]
	steps_VF = index_VF * freqs[None,:] * rad_per_win
	series_VF = jnp.empty((window_size, 2 * F), dtype=jnp.float32)
	series_VF = series_VF.at[:,::2].set(jnp.cos(steps_VF))
	series_VF = series_VF.at[:,1::2].set(jnp.sin(steps_VF))
	return series_VF

class SwiGLU(eqx.Module):
	"""
	Implementation of SwiGLU from https://arxiv.org/pdf/2002.05202 equation 6
	GLU Variants Improve Transformer by N. Shazeer 
	"""
	w: eqx.nn.Linear
	v: eqx.nn.Linear
	w2: eqx.nn.Linear

	def __init__(
		self,
		model_dim: int,
		hidden_dim: int,
		key: PRNGKeyArray,
	):
		k1, k2, k3 = jax.random.split(key, num=3)
		self.w = eqx.nn.Linear(model_dim, hidden_dim, use_bias=False, key=k1)
		self.v = eqx.nn.Linear(model_dim, hidden_dim, use_bias=False, key=k2)
		self.w2 = eqx.nn.Linear(hidden_dim, model_dim, use_bias=False, key=k3)

	def _swish(self, x_M):
		return x_M * jax.nn.sigmoid(x_M)

	def __call__(self, x_M: jax.Array) -> jax.Array:
		w = self.w(x_M)
		v = self.v(x_M)
		s = w * jax.nn.sigmoid(w)
		return self.w2(s * v)

class MLP(eqx.Module):
	"""
	Implement a plain MLP
	"""
	l1: eqx.nn.Linear
	l2: eqx.nn.Linear

	def __init__(self, model_dim: int, hidden_dim: int, key: PRNGKeyArray):
		k1, k2 = jax.random.split(key)
		self.l1 = eqx.nn.Linear(model_dim, hidden_dim, key=k1)
		self.l2 = eqx.nn.Linear(hidden_dim, model_dim, key=k2)
	
	def __call__(self, x_M):
		x = self.l1(x_M)
		x = jax.nn.gelu(x)
		x = self.l2(x)
		return x

class FixedVecSum(eqx.Module):

	def __init__(self):
		pass

	def __call__(self, x_M: jax.Array):
		x_B2 = x_M.reshape(-1, 2)
		y_B2 = jax.vmap(vec_sum_to_angle_sum)(x_B2)
		return y_B2.flatten()


class Model(eqx.Module):
	V: int = eqx.field(static=True)
	M: int = eqx.field(static=True)
	H: int = eqx.field(static=True)

	embed_VM: jax.Array
	unembed: eqx.nn.Linear
	ffn: SwiGLU | MLP
	norm: eqx.nn.RMSNorm

	def __init__(
			self, 
			key: PRNGKeyArray, 
			vocab_size: int, 
			ffn_dim: int,
			ffn_ty: str,
			freqs: list[int],
			phases: list[int] | None):
		if phases is None:
			phases = (0,) * len(freqs)

		k1, k2, k3 = jax.random.split(key, num=3)
		self.V = vocab_size 
		self.H = ffn_dim 
		self.embed_VM = wheels(jnp.array(freqs), jnp.array(phases), self.V) # non-trainable
		self.M = self.embed_VM.shape[1]
		if ffn_ty == 'swiglu':
			self.ffn = SwiGLU(self.M, self.H, key=k1)
		elif ffn_ty == 'mlp':
			self.ffn = MLP(self.M, self.H, key=k1)
		elif ffn_ty == 'trig':
			self.ffn = FixedVecSum()
		self.norm = eqx.nn.RMSNorm(self.M, use_bias=False)
		self.unembed = eqx.nn.Linear(self.M, self.V, use_bias=False, key=k2)

	def _mock(self, a: jax.Array, b: jax.Array) -> jax.Array:
		return (a + b) % self.V

	def mock(self, a_B: jax.Array, b_B: jax.Array) -> jax.Array:
		return jax.vmap(self._mock)(a_B, b_B)

	def __call__(self, a: jax.Array, b: jax.Array) -> jax.Array:
		# M=model, V=vocab.  embed_VM is a subset of DFT rows and not trainable
		xa = self.embed_VM[a,:]  
		xb = self.embed_VM[b,:]
		x = xa + xb 
		x = self.ffn(x)
		x = self.norm(x)
		x = self.unembed(x)
		# x = jnp.einsum('vm, m -> v', self.embed_VM, x)
		return x

def trainable_spec(model):
	spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
	spec = eqx.tree_at(lambda m: m.embed_VM, spec, replace=False)
	return spec

def loss(model, x, y):
	pred_y = jax.vmap(model)(x[:,0], x[:,1])
	xent = optax.softmax_cross_entropy_with_integer_labels(pred_y, y)
	return xent.mean()

def l2norm(diff_model):
	sums = jax.tree_util.tree_map(lambda x: (x ** 2).sum(), diff_model)
	return jax.tree_util.tree_reduce(jnp.add, sums, jnp.array(0.0))


def make_step_fn(optimizer, loss_fn, weight_decay):

	def init(model):
		diff_model, static_model = eqx.partition(model, trainable_spec(model))
		return optimizer.init(diff_model)

	@eqx.filter_jit
	def step(model, opt_state, x, y):
		diff_model, static_model = eqx.partition(model, trainable_spec(model))

		@eqx.filter_value_and_grad
		def compute_loss(diff_model, static_model, x, y):
			model = eqx.combine(diff_model, static_model)
			return loss_fn(model, x, y)

		xent, grads = compute_loss(diff_model, static_model, x, y)
		weight_norm2 = l2norm(diff_model)
		rms_weight = model.norm.weight

		updates, opt_state = optimizer.update(grads, opt_state, diff_model)
		model = eqx.apply_updates(model, updates)
		return xent, weight_norm2, rms_weight, model, opt_state
	return init, step

def vec_sum_to_angle_sum(p: jax.Array):
	"""
	Let p = a + b where:
	a = cos(A), sin(A)
	b = cos(B), sin(B)
	Returns: c = cos(A + B), sin(A + B)
	"""
	u, v = p
	sq = p ** 2
	d = (u * u + v * v)
	x = (u * u - v * v) / d 
	y = (2 * u * v) / d 
	return jnp.where(d == 0.0, jnp.zeros(2), jnp.stack([x, y]))


def trig_ident(a, b):
	"""
	a: (cos(theta), sin(theta))
	b: (cos(phi), sin(phi))
	returns: (cos(theta + phi), sin(theta + phi))
	"""
	ca, sa = a
	cb, sb = b
	return jnp.stack((ca * cb - sa * sb, sa * cb + ca * sb))

def main(
	seed: int, 
	lr: float, 
	weight_decay: float, 
	vocab_size: int, 
	ffn_dim: int, 
	ffn_ty: str,
	freqs: list[int],
	phases: list[int],
	steps: int,
	run_tag: str=None
):
	key = jax.random.key(seed)
	a = jnp.arange(vocab_size)
	b = jnp.arange(vocab_size)
	data = jnp.stack(jnp.meshgrid(a, b), axis=2)
	data = jnp.reshape(data, (-1, data.shape[-1])) 
	model = Model(key, vocab_size, ffn_dim, ffn_ty, freqs, phases)
	target = model.mock(data[:,0], data[:,1])
	tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
	init_fn, step_fn = make_step_fn(tx, loss, weight_decay)
	opt_state = init_fn(model)

	logger_opts = StreamvisOpts(
		flush_every=5.0,
		active=True,
		use_run_handle=None,
		run_tags=None
	)
	
	logger = make_logger(logger_opts)
	logger.start()

	if run_tag is not None:
		logger.add_run_tags(run_tag)

	logger.set_run_attributes(
		arch_resid_dim=model.M,
		opt_wt_decay=weight_decay,
		ffn_ty=ffn_ty
	)

	for i in range(steps+1):
		xent, weight_norm2, rms_weight, model, opt_state = step_fn(model, opt_state, data, target)
		logger.write("metrics", sgd_step=i, lr=lr, xent=xent, top1_acc=0.0, data_split="train", kldiv=0.0) 

		if i % 5000 == 0:
			print(
					f"i: {i}, wn: {weight_norm2.item():4.3f}"
					f", xent: {xent.item():8.6g}",
					f", rms_weight_norm: {(rms_weight ** 2).sum().item():4.3f}"
					)

	logger.stop()

	base_V2 = wheels(jnp.array([1]), jnp.array([0]), vocab_size)
	a_VV2, b_VV2 = jnp.broadcast_arrays(base_V2[:,None], base_V2[None,:])
	sum_VV2 = a_VV2 + b_VV2
	sum_VVM = jnp.tile(sum_VV2, [1,1,len(freqs)])
	V, _, M = sum_VVM.shape
	F = len(freqs)

	trig_out_VV2 = jax.vmap(jax.vmap(trig_ident))(a_VV2, b_VV2)
	ffn_out_VVM = jax.vmap(jax.vmap(model.ffn))(sum_VVM)

	# mean_norm = jnp.sqrt((ffn_out_VVM.reshape(V,V,F,2) ** 2).sum(axis=3)).mean()
	# svg = make_svg_grid(trig_out_VV2, ffn_out_VVM * (mean_norm ** -1))

	unembed_RC2 = model.unembed.weight.reshape(V,M//2,2).transpose(1, 0, 2)
	mean_norm = jnp.sqrt((unembed_RC2 ** 2).sum(axis=2)).mean()

	ref_RC2 = model.embed_VM.reshape(V,M//2,2).transpose(1, 0, 2)
	svg = make_svg_grid(ref_RC2, unembed_RC2 * (mean_norm ** -1))
	file = f"ffn_out.svg"
	with open(file, "w") as f:
		f.write(svg)
		print(f"Wrote {file}, mean_norm={mean_norm:5.4f}")

if __name__ == "__main__":
	fire.Fire(main)
