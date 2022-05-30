try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from PartSeg_measurement.measurement_wrap import measurement

from ._widget import ExampleQWidget, example_magic_widget

__all__ = (
    "__version__",
    "ExampleQWidget",
    "example_magic_widget",
    "measurement",
)
