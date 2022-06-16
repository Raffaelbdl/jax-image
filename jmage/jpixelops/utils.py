import jax.numpy as jnp


def normalize_image(inputs: jnp.ndarray) -> jnp.ndarray:
    """Returns a normalized batch of images

    Args:
        inputs (jnp.ndarray) [N, H, W, C] or [H, W, C] or [H, W]

    Returns:
        new_inputs (jnp.ndarray) [N, H, W, C], scaled to [0, 1]
    """
    in_dtype = inputs.dtype

    if len(inputs.shape) == 2:
        inputs = inputs[..., jnp.newaxis]
    if len(inputs.shape) == 3:
        inputs = inputs[jnp.newaxis]
    if len(inputs.shape) == 4:
        pass
    else:
        raise ValueError(f"inputs should have 2/3/4 dims, not {len(inputs.shape)}")

    if in_dtype == jnp.uint8:
        inputs = jnp.array(inputs, jnp.float32) / 255.0
    else:
        inputs = jnp.clip(inputs, 0.0, 1.0)

    return inputs
