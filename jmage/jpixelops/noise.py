import jax.numpy as jnp
import jax.random as jrng
from jax._src.prng import PRNGKeyArray

from jmage.jpixelops.utils import normalize_image


def normal_noise_process(
    key: PRNGKeyArray, inputs: jnp.ndarray, loc: float, sigma: float
) -> jnp.ndarray:
    """Apply noise to inputs

    Args:
        key (PRNGKeyArray)
        inputs (jnp.ndarray): image format
        loc (float)
        sigma (float)

    Returns:
        outputs (jnp.ndarray): noisy image
    """

    inputs = normalize_image(inputs)

    noise = jrng.normal(key, inputs.shape) * sigma + loc

    outputs = inputs + noise
    outputs = jnp.clip(outputs, 1.0, 0.0)

    return outputs
