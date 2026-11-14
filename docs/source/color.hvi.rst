HVI
===

.. currentmodule:: kornia.color

.. tip::
    Learn more: https://arxiv.org/abs/2502.20272

.. note::
    Hue and saturation fold into the chroma channels H and V, scaled down as the image darkens; the
    intensity I is the HSV value in :math:`[0, 1]`. See :doc:`/get-started/conventions`.

Functions
---------

.. autofunction:: rgb_to_hvi
.. autofunction:: hvi_to_rgb

Modules
-------

.. autoclass:: RgbToHvi
.. autoclass:: HviToRgb
