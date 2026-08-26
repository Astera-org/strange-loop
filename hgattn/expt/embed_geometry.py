"""
Experiments to test how model size and vocab size affect the ability to achieve
near one-hot outputs under a norm constraint
"""

import fire
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray

def main(model_dim: int, vocab_size: int):
    pass



if __name__ == "__main__":
    fire.Fire(main)

