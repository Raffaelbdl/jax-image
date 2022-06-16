from typing import Optional

import jax.numpy as jnp

from jmage.jpixelops.utils import normalize_image


def gamma_process(
    inputs: jnp.ndarray,
    gamma: float,
    masks: Optional[jnp.ndarray] = None,
    mask_indice: Optional[int] = 0,
) -> jnp.ndarray:
    """Apply gamma to inputs

    Args:
        inputs (jnp.ndarray): image format
        gamma (float)
        masks (Optional[jnp.ndarray]): [N, H, W, classes] one hot masks
        mask_indice (Optional[int]): mask to apply gamma

    Returns:
        outputs (jnp.ndarray): image after gamma process
    """
    if gamma <= 0:
        raise ValueError("gamma should be greater than 0")
    if mask_indice > masks.shape[-1]:
        raise IndexError("mask_indice is larger than the number of classes")

    inputs = normalize_image(inputs)

    if masks is None:
        gamma_map = 1 / gamma
    else:
        gamma_map = masks[..., mask_indice] * (1 / gamma)
        gamma_map = gamma_map[..., jnp.newaxis]
        gamma_map = jnp.where(gamma_map == 0.0, 1.0, gamma_map)

    outputs = inputs**gamma_map

    return outputs
