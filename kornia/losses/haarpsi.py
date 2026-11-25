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

import torch
from torch import nn

from kornia import metrics


class HaarPSILoss(nn.Module):
    r"""Compute the HaarPSI dissimilarity loss between two images.

    ``criterion = HaarPSILoss(...)`` stores its arguments, and ``loss = criterion(img1, img2)`` is
    ``1 - haar_psi(img1, img2, ...)`` on two ``(B, C, H, W)`` images: the perceptual similarity of
    :func:`~kornia.metrics.haar_psi` mapped onto a loss, so lower values mean more similar and identical images
    give 0.

    Convention:
        See the Convention block of :func:`~kornia.metrics.haar_psi`.

    Args:
        reduction: ``'mean'`` (default) averages the loss, ``'sum'`` adds it, ``'none'`` keeps one loss per sample.
        data_range: data range of the images, a Python float or int.
        scales: number of Haar wavelet scales, a positive integer.
        subsample: downsample the images by a factor of two first, as the reference does before the wavelet
            decomposition.
        c: similarity stabilising constant.
        alpha: slope of the logistic function applied before the pooling.

    Shape:
        - img1: ``(B, C, H, W)`` with ``C`` in ``{1, 3}``.
        - img2: ``(B, C, H, W)``.
        - Output: a scalar or, for ``reduction='none'``, a ``(B,)`` vector.

    See Also:
        - :func:`~kornia.metrics.haar_psi`: the similarity itself, not the dissimilarity.
        - :class:`~kornia.losses.SSIMLoss`: the structural dissimilarity from a per-pixel SSIM map.

    Examples:
        >>> input1 = torch.rand(1, 1, 32, 32)
        >>> input2 = torch.rand(1, 1, 32, 32)
        >>> criterion = HaarPSILoss()
        >>> loss = criterion(input1, input2)

    """

    def __init__(
        self,
        reduction: str = "mean",
        data_range: float = 1.0,
        scales: int = 3,
        subsample: bool = True,
        c: float = 30.0,
        alpha: float = 4.2,
    ) -> None:
        super().__init__()
        self.reduction: str = reduction
        self.data_range: float = data_range
        self.scales: int = scales
        self.subsample: bool = subsample
        self.c: float = c
        self.alpha: float = alpha

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute the HaarPSI dissimilarity loss between two image batches.

        Args:
            img1: First image batch with shape ``(B, C, H, W)``, where ``C`` is 1 or 3.
            img2: Second image batch with the same shape as ``img1``.

        Returns:
            The loss ``1 - HaarPSI``, reduced according to ``self.reduction``: lower values mean more similar
            images.
        """
        return 1 - metrics.haar_psi(
            img1,
            img2,
            reduction=self.reduction,
            data_range=self.data_range,
            scales=self.scales,
            subsample=self.subsample,
            c=self.c,
            alpha=self.alpha,
        )
