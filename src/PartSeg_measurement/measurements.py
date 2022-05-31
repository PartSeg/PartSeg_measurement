import math
from enum import Enum
from functools import partial
from typing import Literal, Sequence, Union, overload

import numpy as np
import pint

from .measurement_wrap import measurement
from .types import Labels


@overload
def volume(
    labels: np.ndarray,
    voxel_size: Sequence[float],
    per_component: Literal[False] = False,
) -> float:
    ...


@overload
def volume(
    labels: np.ndarray,
    voxel_size: Sequence[float],
    per_component: Literal[True],
) -> np.ndarray:
    ...


@measurement
def volume(
    labels: Labels, voxel_size, per_component: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the volume of the object marked with positive pixels
    on labels array.

    Parameters
    ----------
    labels : Labels
        array with information of object voxels
    voxel_size : (float, ...)
        tuple with voxel size in each dimension
    per_component : bool
        If calculate the volume per component or global
    """
    if per_component:
        return np.bincount(labels.flat)[1:] * np.prod(voxel_size)
    return np.count_nonzero(labels) * math.prod(voxel_size)


@overload
def voxels(labels: np.ndarray, per_component: Literal[False] = False) -> float:
    ...


@overload
def voxels(labels: np.ndarray, per_component: Literal[True]) -> np.ndarray:
    ...


@measurement
def voxels(
    labels: Labels, per_component: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the number of voxels of the object marked with positive
    pixels on labels array.

    Parameters
    ----------
    labels : Labels
        array with information of object voxels
    per_component :  bool
        If calculate the number of voxels per component or global
    """
    reg = pint.get_application_registry()
    if per_component:
        return np.bincount(labels.flat)[1:] * reg.pixel
    return np.count_nonzero(labels) * reg.pixel


class PixelOpEnum(Enum):
    """
    Enum for pixel aggregation operations
    """

    sum = partial(np.sum)
    mean = partial(np.mean)
    median = partial(np.median)
    max = partial(np.max)
    min = partial(np.min)
    std = partial(np.std)


@measurement
def pixel_brightness(
    labels: Labels, image: np.ndarray, operation: PixelOpEnum
) -> float:
    """
    Calculate the sum of the pixel brightness of the object marked
    with positive pixels on labels array.

    Parameters
    ----------
    labels : Labels
        array with information of object voxels
    image : Image
        array with pixel brightness
    operation : PixelOpEnum
        operation to be applied to the pixel brightness
    """
    return (
        operation.value(image[labels > 0]) if np.any(labels) else 0
    ) * pint.get_application_registry().lux
