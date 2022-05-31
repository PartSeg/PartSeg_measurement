# PartSeg_measurement

[![License: BSD3](https://img.shields.io/pypi/l/PartSeg_measurement.svg?color=green)](https://github.com/czaki/PartSeg_measurement/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PartSeg_measurement.svg?color=green)](https://pypi.org/project/PartSeg_measurement)
[![Python Version](https://img.shields.io/pypi/pyversions/PartSeg_measurement.svg?color=green)](https://python.org)
[![tests](https://github.com/czaki/PartSeg_measurement/workflows/tests/badge.svg)](https://github.com/czaki/PartSeg_measurement/actions)
[![codecov](https://codecov.io/gh/czaki/PartSeg_measurement/branch/main/graph/badge.svg)](https://codecov.io/gh/czaki/PartSeg_measurement)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/PartSeg_measurement)](https://napari-hub.org/plugins/PartSeg_measurement)
[![Documentation Status](https://readthedocs.org/projects/partseg-measurement/badge/?version=latest)](https://partseg-measurement.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Czaki/PartSeg_measurement/main.svg)](https://results.pre-commit.ci/latest/github/Czaki/PartSeg_measurement/main)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/Czaki/PartSeg_measurement.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Czaki/PartSeg_measurement/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Czaki/PartSeg_measurement.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Czaki/PartSeg_measurement/context:python)

Measurement engine for imaging data from PartSeg

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

You can install `PartSeg_measurement` via [pip]:

    pip install PartSeg_measurement



To install latest development version :

    pip install git+https://github.com/czaki/PartSeg_measurement.git


## Plugin Widget

TODO

## Library

```python
import numpy as np
import tifffile

from PartSeg_measurement import measurement, Image, Labels

@measurement
def volume(labels: Labels, voxel_size):
   """
   Calculate the volume of the object marked with positive pixels on labels array.
   """
   return np.count_nonzero(labels) * np.prod(voxel_size)

@measurement
def sum_of_pixel_brightness(labels: Labels, image: Image) -> float:
   """
   Calculate the sum of the pixel brightness of the object marked with positive pixels on labels array.
   """
   return np.sum(image[labels > 0]) if np.any(labels) else 0

density = sum_of_pixel_brightness / volume

image = tifffile.imread('image.tif')
labels = tifffile.imread('labels.tif')

print(f"Density: {density(labels=labels, image=image, voxel_size=(210, 70, 70))}")

```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"PartSeg_measurement" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/czaki/PartSeg_measurement/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
