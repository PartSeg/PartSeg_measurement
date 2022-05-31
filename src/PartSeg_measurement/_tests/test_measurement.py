import numpy as np
import pint
import pytest

from PartSeg_measurement.measurement_wrap import MeasurementFunctionWrap
from PartSeg_measurement.measurements import (
    PixelOpEnum,
    pixel_brightness,
    volume,
    voxels,
)


@pytest.fixture
def units():
    return pint.get_application_registry()


@pytest.fixture
def image():
    data = np.zeros((10, 10, 10))
    data[2:-2, 2:-2, 2:-2] = 10
    data[3:-3, 3:-3, 3:-3] = 20
    data[4:-4, 4:-4, 4:-4] = 30
    return data


@pytest.fixture
def labels(image):
    return (image > 15).astype(np.uint8)


def mask(image):
    return (image > 0).astype(np.uint8)


class TestVolume:
    def test_type(self):
        assert isinstance(volume, MeasurementFunctionWrap)

    def test_call(self, labels, units):
        assert volume(labels=labels, voxel_size=(1, 1, 1)) == 64
        assert volume(labels=labels, voxel_size=(2, 1, 1)) == 128
        assert (
            volume(
                labels=labels,
                voxel_size=(2 * units.nm, 1 * units.nm, 1 * units.nm),
            )
            == 128 * units.nm**3
        )


class TestVoxels:
    def test_type(self):
        assert isinstance(voxels, MeasurementFunctionWrap)

    def test_call(self, labels, units):
        assert voxels(labels=labels) == 64 * units.pixel


class TestPixelBrightness:
    def test_type(self):
        assert isinstance(pixel_brightness, MeasurementFunctionWrap)

    @pytest.mark.parametrize("op", PixelOpEnum.__members__.values())
    def test_call(self, image, labels, op, units):
        assert (
            pixel_brightness(image=image, labels=labels, operation=op)
            == op.value(image[labels > 0]) * units.lux
        )
