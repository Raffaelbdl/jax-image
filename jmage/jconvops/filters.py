"""Adapted from tensorflow addons

https://github.com/tensorflow/addons/blob/b2dafcfa74c5de268b8a5c53813bc0b89cadf386/tensorflow_addons/image/filters.py#L203
"""
import functools
from typing import Iterable, Tuple, Union

import jax
import jax.lax as lax
import jax.nn as nn
import jax.numpy as jnp

from jconvops._src.utils import normalize_kernel, normalize_iterable


def in_filter2d(
    inputs: jnp.ndarray,
):
    inputs = jnp.array(inputs)
    img_dtype = inputs.dtype
    dtype = jnp.float32
    if img_dtype == jnp.uint8:
        inputs = jnp.array(inputs, dtype) / 255.0

    if len(inputs.shape) == 2:
        inputs = inputs[jnp.newaxis, ..., jnp.newaxis]
    elif len(inputs.shape) == 3:
        inputs = inputs[jnp.newaxis]

    return inputs, img_dtype, dtype


def out_filter2d(outputs: jnp.ndarray, img_dtype):
    if img_dtype == jnp.uint8:
        outputs = outputs * 255.0
        outputs = jnp.array(outputs, img_dtype)
    return outputs


def _apply_kernel_depthwise(inputs: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """

    Args:
        inputs (Array) [N, H, W, C]
        kernel (Array) [K1, K2]
    """
    kernel = normalize_kernel(kernel)
    inputs = jnp.transpose(inputs, (0, 3, 1, 2))
    inputs = inputs[:, :, jnp.newaxis]
    outputs = jax.vmap(
        functools.partial(lax.conv, rhs=kernel, window_strides=(1, 1), padding="SAME"),
        in_axes=(1),
    )(inputs)
    outputs = outputs[:, :, 0]
    outputs = jnp.transpose(outputs, (1, 2, 3, 0))
    return outputs


def _get_gaussian_kernel(sigma: float, filter_shape: int) -> jnp.ndarray:
    x = jnp.arange(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = -(x**2) / (2 * sigma**2)
    x = nn.softmax(x)
    return x


def _get_gaussian_kernel_2d(
    gaussian_filter_x: jnp.ndarray, gaussian_filter_y: jnp.ndarray
) -> jnp.ndarray:
    return jnp.dot(
        gaussian_filter_x[..., jnp.newaxis],
        jnp.transpose(gaussian_filter_y[..., jnp.newaxis]),
    )


def gaussian_filter2d(
    inputs: jnp.ndarray,
    filter_shape: Union[int, Iterable[int]] = (3, 3),
    sigma: Union[float, Iterable[float]] = 1.0,
) -> jnp.ndarray:

    inputs, img_dtype, dtype = in_filter2d(inputs)

    sigma = normalize_iterable(sigma, 2, dtype)
    if any(s < 0 for s in sigma):
        raise ValueError("sigma should be greater than or equal to 0")
    filter_shape = normalize_iterable(filter_shape, 2, jnp.int32)

    gaussian_kernel_x = _get_gaussian_kernel(sigma[0], filter_shape[0])
    gaussian_kernel_y = _get_gaussian_kernel(sigma[1], filter_shape[1])
    gaussian_kernel = _get_gaussian_kernel_2d(gaussian_kernel_x, gaussian_kernel_y)

    outputs = _apply_kernel_depthwise(inputs, gaussian_kernel)
    outputs = out_filter2d(outputs, img_dtype)

    return outputs


def _get_mean_kernel(filter_shape: Tuple[int]) -> jnp.ndarray:
    x = jnp.ones(filter_shape)
    x = x / jnp.prod(jnp.array(filter_shape))
    return x


def mean_filter2d(
    inputs: jnp.ndarray,
    filter_shape: Union[int, Iterable[int]] = (3, 3),
) -> jnp.ndarray:

    inputs, img_dtype, dtype = in_filter2d(inputs)
    filter_shape = normalize_iterable(filter_shape, 2, jnp.int32)

    mean_kernel = _get_mean_kernel(filter_shape)

    outputs = _apply_kernel_depthwise(inputs, mean_kernel)
    outputs = out_filter2d(outputs, img_dtype)

    return outputs
