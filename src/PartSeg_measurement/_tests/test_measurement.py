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
    data = np.zeros((10, 10, 20))
    data[2:-2, 2:-2, 2:-2] = 10
    data[3:-3, 3:-3, 3:-3] = 20
    data[4:-4, 4:-4, 4:-14] = 30
    return data


@pytest.fixture
def labels(image):
    res = (image > 15).astype(np.uint8)
    res[3:-3, 3:-3, 10:-3] = 2
    return res


def mask(image):
    return (image > 0).astype(np.uint8)


class TestVolume:
    def test_type(self):
        assert isinstance(volume, MeasurementFunctionWrap)

    def test_call(self, labels, units):
        assert volume(labels=labels, voxel_size=(1, 1, 1)) == 224
        assert volume(labels=labels, voxel_size=(2, 1, 1)) == 448
        assert (
            volume(
                labels=labels,
                voxel_size=(2 * units.nm, 1 * units.nm, 1 * units.nm),
            )
            == 448 * units.nm**3
        )
        assert (
            volume(
                labels=labels,
                voxel_size=np.array([2, 1, 1]) * units.nm,
            )
            == 448 * units.nm**3
        )

    def test_per_component_call(self, labels, units):
        assert list(
            volume(labels=labels, voxel_size=(1, 1, 1), per_component=True)
        ) == [112, 112]
        assert list(
            volume(
                labels=labels,
                voxel_size=np.array((2, 2, 1)) * units.nm,
                per_component=True,
            )
        ) == [448 * units.nm**3, 448 * units.nm**3]


class TestVoxels:
    def test_type(self):
        assert isinstance(voxels, MeasurementFunctionWrap)

    def test_call(self, labels, units):
        assert voxels(labels=labels) == 224 * units.pixel

    def test_call_per_component(self, labels, units):
        assert list(voxels(labels=labels, per_component=True)) == [
            112 * units.pixel,
            112 * units.pixel,
        ]


class TestPixelBrightness:
    def test_type(self):
        assert isinstance(pixel_brightness, MeasurementFunctionWrap)

    @pytest.mark.parametrize("op", PixelOpEnum.__members__.values())
    def test_call(self, image, labels, op, units):
        assert (
            pixel_brightness(image=image, labels=labels, operation=op)
            == op.value(image[labels > 0]) * units.lux
        )
