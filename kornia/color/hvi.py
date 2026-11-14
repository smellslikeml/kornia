# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn

from .hsv import hsv_to_rgb, rgb_to_hsv


def rgb_to_hvi(image: torch.Tensor, density_k: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to the HVI color space.

    ``hvi = rgb_to_hvi(image)`` converts a ``(*, 3, H, W)`` RGB image in :math:`[0, 1]` to HVI, the color space of
    `HVI: A New Color Space for Low-light Image Enhancement <https://arxiv.org/abs/2502.20272>`__ (CVPR 2025).

    HVI is HSV made polar for the dark: the intensity channel ``I`` is the HSV value, and the hue and saturation
    fold into the chroma channels ``H = cs * S * cos(hue)`` and ``V = cs * S * sin(hue)``, scaled by the color
    sensitivity ``cs = (sin(I * pi / 2) + eps) ** density_k``. The scale shrinks the chroma disk as the image
    darkens, which evens out the color distribution a low-light enhancement network sees.

    Convention:
        Channels are H, V, I at axis -3. ``I`` is in :math:`[0, 1]` and the chroma channels lie within a disk of
        radius about one, ``[-1, 1]`` each. Hue follows :func:`rgb_to_hsv`: radians in :math:`[0, 2\pi)`, zero at
        grayscale. The result is not clipped.

    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        density_k: exponent ``k`` of the color sensitivity. Larger values shrink the chroma of dark pixels
            further; the reference default is ``0.2``.
        eps: added to the HSV value in the saturation denominator and inside the color sensitivity, keeping the
            fractional power, and its gradient, non-zero at black.

    Returns:
        HVI image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`hvi_to_rgb`: the inverse.
        - :class:`RgbToHvi`: the module form, with ``density_k`` and ``eps`` stored.
        - :func:`rgb_to_hsv`: the underlying decomposition.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hvi(input)  # 2x3x4x5

    """
    hue, saturation, value = torch.unbind(rgb_to_hsv(image, eps), dim=-3)

    # The color-sensitive scale shrinks the chroma disk as the value darkens; eps keeps the base of the
    # fractional power, and its gradient, non-zero at black.
    color_sensitive = (torch.sin(value * (math.pi / 2.0)) + eps).pow(density_k)

    return torch.stack(
        (color_sensitive * saturation * torch.cos(hue), color_sensitive * saturation * torch.sin(hue), value),
        dim=-3,
    )


def hvi_to_rgb(image: torch.Tensor, density_k: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from the HVI color space to RGB.

    ``rgb = hvi_to_rgb(image)`` converts a ``(*, 3, H, W)`` HVI image to RGB: the chroma channels are divided by
    the color sensitivity recomputed from ``I``, and read back as the polar coordinates of hue and saturation,
    ``hue = atan2(V + eps, H + eps)`` and ``S = sqrt(H ** 2 + V ** 2 + eps)``, before :func:`hsv_to_rgb` applies.

    Convention:
        Channels are H, V, I at axis -3; see :func:`rgb_to_hvi`. The eps terms hold hue and saturation defined,
        with finite gradients, where the chroma vanishes, at the cost of a saturation floor of ``sqrt(eps)``: a
        :func:`rgb_to_hvi` / :func:`hvi_to_rgb` round trip moves grayscale pixels by up to ``V * sqrt(eps)``, about
        ``5e-5``. The result is not clipped, so out-of-range chroma or intensity can give RGB values outside
        :math:`[0, 1]`.

    Args:
        image: HVI image with shape ``(*, 3, H, W)``.
        density_k: exponent ``k`` of the color sensitivity; must match the encode.
        eps: added inside the color sensitivity, the atan2 arguments and the square root, as in the encode.

    Returns:
        RGB image with shape ``(*, 3, H, W)``.

    See Also:
        - :func:`rgb_to_hvi`: the inverse.
        - :class:`HviToRgb`: the module form, with ``density_k`` and ``eps`` stored.
        - :func:`hsv_to_rgb`: the final reconstruction.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hvi_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    chroma_h, chroma_v, value = torch.unbind(image, dim=-3)

    color_sensitive = (torch.sin(value * (math.pi / 2.0)) + eps).pow(density_k)
    # Guard black pixels before division: eps can underflow in float16, including in backward, so the
    # color sensitivity at I = 0 divides by zero there. The encode leaves their chroma zero, which the
    # substitute divisor of one keeps.
    divisor = color_sensitive + eps
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

    # Divide out the encode's scale, then read hue and saturation back from the polar coordinates. The eps
    # terms keep both defined, with finite gradients, where the chroma vanishes (grayscale pixels).
    chroma_h = chroma_h / divisor
    chroma_v = chroma_v / divisor
    hue = torch.atan2(chroma_v + eps, chroma_h + eps)
    saturation = torch.sqrt(chroma_h * chroma_h + chroma_v * chroma_v + eps)

    return hsv_to_rgb(torch.stack((hue, saturation, value), dim=-3))


class RgbToHvi(nn.Module):
    r"""Convert an image from RGB to HVI (module form).

    ``hvi = RgbToHvi(density_k, eps)`` stores the arguments, and ``out = hvi(image)`` is :func:`rgb_to_hvi` with
    them.

    Convention:
        - See the convention block of :func:`rgb_to_hvi`.

    Args:
        density_k: exponent ``k`` of the color sensitivity.
        eps: added to the HSV value in the saturation denominator and inside the color sensitivity.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`rgb_to_hvi`: the one-call form.
        - :class:`HviToRgb`: the inverse.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hvi = RgbToHvi()
        >>> output = hvi(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, density_k: float = 0.2, eps: float = 1e-8) -> None:
        super().__init__()
        self.density_k = density_k
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB tensor to HVI.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` is the channel dimension, and ``H``/``W`` are height and width.

        Returns:
            HVI tensor with shape ``(*, 3, H, W)``.
        """
        return rgb_to_hvi(image, self.density_k, self.eps)


class HviToRgb(nn.Module):
    r"""Convert an image from HVI to RGB (module form).

    ``out = HviToRgb(density_k, eps)(image)`` is :func:`hvi_to_rgb` with the arguments stored.

    Convention:
        - See the convention block of :func:`hvi_to_rgb`.

    Args:
        density_k: exponent ``k`` of the color sensitivity; must match the encode.
        eps: added inside the color sensitivity, the atan2 arguments and the square root.

    Shape:
        - image: ``(*, 3, H, W)``
        - output: ``(*, 3, H, W)``

    See Also:
        - :func:`hvi_to_rgb`: the one-call form.
        - :class:`RgbToHvi`: the inverse.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HviToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def __init__(self, density_k: float = 0.2, eps: float = 1e-8) -> None:
        super().__init__()
        self.density_k = density_k
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an HVI tensor to RGB.

        Args:
            image: Input tensor with shape ``(*, 3, H, W)``.
                Here, ``*`` means any number of leading dimensions (for example, batch size),
                ``3`` is the channel dimension, and ``H``/``W`` are height and width.

        Returns:
            RGB tensor with shape ``(*, 3, H, W)``.
        """
        return hvi_to_rgb(image, self.density_k, self.eps)
