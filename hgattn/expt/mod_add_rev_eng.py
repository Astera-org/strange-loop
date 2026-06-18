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

def make_svg_grid(
		ref_VV2: jax.Array,
		arrows_VV2: jax.Array,
		margin: int=20,
		arrow_frac: float=0.8,
		stroke: float=2.0,
		cell: int=60,
		) -> str:
	"""
	Given a f4[V,V,2], create an SVG image with arrows arranged in a grid
	"""
	n = arrows_VV2.shape[0]
	arrows_VV2 = np.array(arrows_VV2)
	ref_VV2 = np.array(ref_VV2)

	size = n * cell + 2 * margin
	L = arrow_frac * cell          # on-screen length of a "unit" arrow
	half = L / 2.0

	# Arrowhead geometry (drawn via a reusable marker).
	head_len = 0.30 * L
	head_w = 0.22 * L

	out = []
	out.append(
			f'<svg xmlns="http://www.w3.org/2000/svg" '
			f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
			)
	out.append(f'  <rect width="{size}" height="{size}" fill="white"/>')

	# One marker, auto-oriented along each line, sized in user units.
	out.append('  <defs>')
	out.append(
			f'    <marker id="head" markerUnits="userSpaceOnUse" '
			f'markerWidth="{head_len}" markerHeight="{head_w}" '
			f'refX="{head_len}" refY="{head_w / 2}" orient="auto">'
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

	def draw_one(i, j, data_VV2, color):
		cx = margin + (j + 0.5) * cell
		cy = margin + (i + 0.5) * cell
		theta = i * j * 2.0 * math.pi / n
		dx = data_VV2[i,j,0] 
		dy = data_VV2[i,j,1] 
		# SVG's y-axis points downward, so negate dy to keep the
		# angle counter-clockwise from the positive x-axis on screen.
		tail_x = cx - half * dx
		tail_y = cy + half * dy
		tip_x = cx + half * dx
		tip_y = cy - half * dy
		out.append(
				f'    <line x1="{tail_x:.3f}" y1="{tail_y:.3f}" '
				f'x2="{tip_x:.3f}" y2="{tip_y:.3f}" stroke="{color}"/>'
				)

	for i in range(n):
		for j in range(n):
			draw_one(i, j, arrows_VV2 * 2.0, "red")
			draw_one(i, j, ref_VV2, "black")

	out.append('  </g>')
	out.append('</svg>')
	return '\n'.join(out)



def wheels(freqs: jax.Array, window_size: int) -> jax.Array:
	"""
	Produce a fourier series with frequencies freqs * 2 \\pi / window_size
	"""
	F = freqs.shape[0]
	rad_per_win = 2.0 * jnp.pi / window_size
	steps = jnp.arange(window_size)[None,:] * freqs[:,None] * rad_per_win
	series = jnp.empty((2 * F, window_size), dtype=jnp.float32)
	series = series.at[::2,:].set(jnp.cos(steps))
	series = series.at[1::2,:].set(jnp.sin(steps))
	return series


class Model(eqx.Module):
	V: int = eqx.field(static=True)
	M: int = eqx.field(static=True)
	H: int = eqx.field(static=True)

	embed_MV: jax.Array
	f1: eqx.nn.Linear
	f2: eqx.nn.Linear

	def __init__(
			self, 
			key: PRNGKeyArray, 
			vocab_size: int, 
			ffn_dim: int,
			freqs: list[int]):
		k1, k2 = jax.random.split(key)
		self.V = vocab_size 
		self.H = ffn_dim 
		self.embed_MV = wheels(jnp.array(freqs), self.V) # non-trainable
		self.M = self.embed_MV.shape[0]
		self.f1 = eqx.nn.Linear(self.M, self.H, key=k1)
		self.f2 = eqx.nn.Linear(self.H, self.M, key=k2)


	def mlp(self, x_M: jax.Array) -> jax.Array:
		x = self.f1(x_M)
		x = jax.nn.gelu(x)
		x = self.f2(x)
		return x

	def _mock(self, a: jax.Array, b: jax.Array) -> jax.Array:
		return (a + b) % self.V

	def mock(self, a_B: jax.Array, b_B: jax.Array) -> jax.Array:
		return jax.vmap(self._mock)(a_B, b_B)

	def __call__(self, a: jax.Array, b: jax.Array) -> jax.Array:
		"""
		Perform modular addition between a_V and b_V encoded as one-hot integers
		"""
		xa = self.embed_MV[:,a]
		xb = self.embed_MV[:,b]
		x = xa + xb 
		x = self.mlp(x)
		x = jnp.einsum('mv, m -> v', self.embed_MV, x)
		return x

def trainable_spec(model):
	spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
	spec = eqx.tree_at(lambda m: m.embed_MV, spec, replace=False)
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

		updates, opt_state = optimizer.update(grads, opt_state, diff_model)
		model = eqx.apply_updates(model, updates)
		return xent, weight_norm2, model, opt_state
	return init, step

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
	freqs: list[int],
	steps: int
):
	key = jax.random.key(seed)
	a = jnp.arange(vocab_size)
	b = jnp.arange(vocab_size)
	data = jnp.stack(jnp.meshgrid(a, b), axis=2)
	data = jnp.reshape(data, (-1, data.shape[-1])) 
	model = Model(key, vocab_size, ffn_dim, freqs)
	target = model.mock(data[:,0], data[:,1])
	tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
	init_fn, step_fn = make_step_fn(tx, loss, weight_decay)
	opt_state = init_fn(model)

	for i in range(steps):
		xent, weight_norm2, model, opt_state = step_fn(model, opt_state, data, target)
		if i % 5000 == 0:
			print(f"i: {i}, wn: {weight_norm2.item():4.3f}, xent: {xent.item():8.6g}")

	base_2V = wheels(jnp.array([1]), vocab_size)
	base_V2 = jnp.transpose(base_2V, (1, 0))
	a_VV2, b_VV2 = jnp.broadcast_arrays(base_V2[:,None], base_V2[None,:])
	sum_VV2 = a_VV2 + b_VV2
	trig_fn = jax.vmap(jax.vmap(trig_ident))

	sum_VVM = jnp.repeat(sum_VV2, len(freqs), axis=2)

	sum_BM = sum_VVM.reshape(-1, sum_VVM.shape[2])
	B, M = sum_BM.shape
	V = model.V
	trig_out_VV2 = trig_fn(a_VV2, b_VV2)
	mlp_out_BM = jax.vmap(model.mlp)(sum_BM)
	mlp_out_VVM = jnp.reshape(mlp_out_BM, (V, V, M))

	for offset in range(0, M, 2):
		mlp_out_VV2 = mlp_out_VVM[:,:,offset:offset+2]
		mean_norm = jnp.sqrt((mlp_out_VV2 ** 2).sum(axis=2)).mean()
		svg = make_svg_grid(trig_out_VV2, mlp_out_VV2 * (mean_norm ** -1))
		file = f"mlp_out_{offset}.svg"
		with open(file, "w") as f:
			f.write(svg)
			print(f"Wrote {file}")

if __name__ == "__main__":
	fire.Fire(main)
