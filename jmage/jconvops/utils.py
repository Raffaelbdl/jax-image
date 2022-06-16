from typing import Iterable, Tuple

import jax.numpy as jnp


def normalize_iterable(value, n, dtype) -> Tuple[jnp.ndarray]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError("sigma should be a float or a list/tuple of 2 floats")
        else:
            value = tuple([jnp.array(v, dtype) for v in value])
    else:
        value = (jnp.array(value, dtype=dtype),) * n
    return value


def normalize_kernel(kernel: jnp.ndarray) -> jnp.ndarray:
    return kernel[jnp.newaxis, jnp.newaxis]
