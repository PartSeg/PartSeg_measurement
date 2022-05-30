"""
Types corresponding to name convention from napari.
"""

from typing import TypeVar

import numpy as np

Labels = TypeVar("Labels", bound=np.ndarray)
Image = TypeVar("Image", bound=np.ndarray)
