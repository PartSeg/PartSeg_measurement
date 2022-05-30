.. PartSeg measurement documentation master file, created by
   sphinx-quickstart on Thu May 26 09:41:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PartSeg measurement's documentation!
===============================================

``PartSeg_measurement`` is extraction and generalization of the
:py:mod:`PartSeg` measurement engine for napari imaging data.

Example of usage:

.. code-block:: python

   import tifffile
   from PartSeg_measurement import measurement

   @measurement(units="nm**3")
   def volume(labels: Labels, voxel_size):
       """
       Calculate the volume of the object marked with positive pixels on labels array.
       """
       return np.count_nonzero(labels) * np.prod(voxel_size)

   @measurement(units="brightness")
   def sum_of_pixel_brightness(labels: Labels, image: np.ndarray) -> float:
       """
       Calculate the sum of the pixel brightness of the object marked with positive pixels on labels array.
       """
       return np.sum(image[labels > 0]) if np.any(labels) else 0

   density = sum_of_pixel_brightness / volume

   image = tifffile.imread('image.tif')
   labels = tifffile.imread('labels.tif')

   print(f"Density: {density(labels=labels, image=image, voxel_size=(210, 70, 70))}")


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   measurement_wrap



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
