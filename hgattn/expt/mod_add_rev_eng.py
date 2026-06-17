"""
Reverse engineered network for modular addition
"""
import fire
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import optax


def wheels(freqs: jax.Array, window_size: int) -> jax.Array:
    """
    Produce a fourier series with frequencies freqs * 2 \\pi / window_size
    """
    F = freqs.shape[0]
    rad_per_win = 2.0 * jnp.pi / window_size
    steps = jnp.arange(window_size)[None,:] * freqs[:,None] * rad_per_win
    series = jnp.empty((2 * F, window_size), dtype=jnp.float32)
    series = series.at[::2,:].set(jnp.sin(steps))
    series = series.at[1::2,:].set(jnp.cos(steps))
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

def loss(model, x, y):
    pred_y = jax.vmap(model)(x[:,0], x[:,1])
    xent = optax.softmax_cross_entropy_with_integer_labels(pred_y, y)
    return xent.mean()


def make_step_fn(optimizer, loss_fn):

    def trainable_spec(model):
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
        spec = eqx.tree_at(lambda m: m.embed_MV, spec, replace=False)
        return spec

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
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        model = eqx.apply_updates(model, updates)
        return xent, model, opt_state
    return init, step

def main(seed: int, lr: float, vocab_size: int, ffn_dim: int, freqs: list[int]):
    key = jax.random.key(seed)
    a = jnp.arange(vocab_size)
    b = jnp.arange(vocab_size)
    data = jnp.stack(jnp.meshgrid(a, b), axis=2)
    data = jnp.reshape(data, (-1, data.shape[-1])) 
    model = Model(key, vocab_size, ffn_dim, freqs)
    target = model.mock(data[:,0], data[:,1])
    tx = optax.adamw(learning_rate=lr)
    init_fn, step_fn = make_step_fn(tx, loss)
    opt_state = init_fn(model)

    for i in range(10001):
        xent, model, opt_state = step_fn(model, opt_state, data, target)
        if i % 1000 == 0:
            print(f"i: {i}, xent: {xent.item():8.6f}")

    # Now, show the actual function implemented by the learned MLP



if __name__ == "__main__":
    fire.Fire(main)
