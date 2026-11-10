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

from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_IS_TENSOR,
    KORNIA_CHECK_SAME_DEVICE,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
)


def _spectrum_patches(x: torch.Tensor, patch_factor: int) -> torch.Tensor:
    """Orthonormal 2-D DFT of a ``patch_factor``-by-``patch_factor`` grid of patches of ``x``.

    The grid is read row by row, so patch ``i * patch_factor + j`` is the patch at row ``i`` and
    column ``j`` of the grid, and the returned spectrum is complex with shape
    ``(B, patch_factor**2, C, H // patch_factor, W // patch_factor)``.
    """
    batch, channels, height, width = x.shape
    patch_h, patch_w = height // patch_factor, width // patch_factor
    # One reshape instead of the patch-by-patch loop of the reference implementation; both read the
    # same (row i, column j) patch into the same slot.
    patches = x.reshape(batch, channels, patch_factor, patch_h, patch_factor, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, patch_factor * patch_factor, channels, patch_h, patch_w)
    return torch.fft.fft2(patches, norm="ortho")


def focal_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    patch_factor: int = 1,
    ave_spectrum: bool = False,
    log_matrix: bool = False,
    batch_matrix: bool = False,
    matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Weight the 2-D DFT distance between two images by its own magnitude.

    ``loss = focal_frequency_loss(pred, target)`` takes the orthonormal 2-D discrete Fourier
    transform (DFT) of each patch of ``pred`` and ``target``, measures the squared distance
    :math:`d` between the two spectra per frequency, and averages :math:`d` weighted by a weight
    :math:`w` that itself grows with :math:`d`. The frequencies a prediction currently gets hardest
    therefore dominate the loss, and the ones it already matches fade out. Use
    :func:`~kornia.losses.ssim_loss` for a reconstruction loss in the spatial domain instead.

    .. math::

        \Delta &= \operatorname{DFT}_{ortho}(pred) - \operatorname{DFT}_{ortho}(target) \\

        d &= \Delta_{re}^{2} + \Delta_{im}^{2} \\

        \tilde{w} &= \begin{cases} \log\left(\sqrt{d}^{\alpha} + 1\right) & \text{if log\_matrix} \\
        \sqrt{d}^{\alpha} & \text{otherwise} \end{cases} \\

        w &= \operatorname{detach}\left(\operatorname{clamp}\left(\frac{\tilde{w}}{\max \tilde{w}}, 0, 1\right)\right) \\

        \text{loss} &= \operatorname{mean}(w \, d)

    Reference:
        [1] https://arxiv.org/abs/2012.12821
        [2] https://github.com/EndlessSora/focal-frequency-loss (MIT), reimplemented here.

    Convention:
        - The weight is detached, as in the reference: the gradient flows through the distance
          :math:`d` only, so the backward pass reweights the frequencies rather than differentiating
          the reweighting. A loss whose weight is frozen at its current value has exactly this
          gradient, the full composite does not.
        - The distance is :math:`\Delta_{re}^{2} + \Delta_{im}^{2}`, not :math:`|\Delta|^{2}`, whose
          gradient is undefined where the spectra agree.
        - ``max`` normalizes the weight per image, patch and channel over the frequency grid, or over
          the whole batch and every patch and channel with ``batch_matrix=True``.
        - Identical inputs give a weight of ``0/0``, which ``nan_to_num`` turns into the zero weight
          and a loss of exactly 0.
        - The loss is a scalar mean over the batch, the patches and the frequencies. There is no
          ``reduction`` argument; :class:`~kornia.losses.FocalFrequencyLoss` adds the only multiplier.
        - Half-precision inputs are transformed and averaged in float32, gradients included, and the
          scalar is returned in the input dtype.
        - Despite the shared word, this is not the classification loss
          :func:`~kornia.losses.focal_loss`: no logits, no class weighting, and a reconstruction
          target instead.

    Args:
        pred: prediction with shape ``(B, C, H, W)``.
        target: target with the same shape as ``pred``.
        alpha: exponent of the focal weight; a larger value focuses the loss harder on the few
            worst frequencies.
        patch_factor: number of patches per side. The image is split into ``patch_factor ** 2``
            patches, each transformed on its own, so a local spectrum is compared per patch.
        ave_spectrum: average the two spectra over the batch before the distance, which compares a
            batch-average spectrum instead of each image's own.
        log_matrix: take the logarithm of the weight before it is normalized, which evens out a few
            dominant frequencies.
        batch_matrix: normalize the weight by its maximum over the whole batch instead of per image,
            patch and channel.
        matrix: a predefined weight with a shape that broadcasts against the per-frequency distance,
            used as given and detached, instead of the dynamic weight above.

    Returns:
        the loss, a scalar.

    Raises:
        BaseError: if either input is not a tensor with shape ``(B, C, H, W)``, if their shapes or
            devices differ, or if ``patch_factor`` does not divide ``H`` and ``W``.

    See Also:
        - :class:`~kornia.losses.FocalFrequencyLoss`: the module form.
        - :func:`~kornia.losses.ssim_loss`: a reconstruction loss in the spatial domain.
        - :func:`~kornia.losses.focal_loss`: the classification loss that shares the word "focal".

    Example:
        >>> pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        >>> target = torch.zeros_like(pred)
        >>> focal_frequency_loss(pred, target)
        tensor(6.7000)
        >>> focal_frequency_loss(pred, target, patch_factor=2)
        tensor(7.5000)

    """
    KORNIA_CHECK_IS_TENSOR(pred)

    KORNIA_CHECK_IS_TENSOR(target)

    KORNIA_CHECK_SHAPE(pred, ["B", "C", "H", "W"])

    KORNIA_CHECK_SAME_SHAPE(pred, target)

    KORNIA_CHECK_SAME_DEVICE(pred, target)

    KORNIA_CHECK(patch_factor > 0, lambda: f"The patch factor must be positive. Got: {patch_factor}")

    KORNIA_CHECK(
        pred.shape[-2] % patch_factor == 0 and pred.shape[-1] % patch_factor == 0,
        lambda: f"The patch factor must divide the image height and width. Got: {patch_factor}, image {pred.shape}",
    )

    # The transform has no float16 kernel on CPU (and no complex half), so compute in float32.
    output_dtype = torch.promote_types(pred.dtype, target.dtype)
    compute_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype

    pred_spectrum = _spectrum_patches(pred.to(compute_dtype), patch_factor)
    target_spectrum = _spectrum_patches(target.to(compute_dtype), patch_factor)

    if ave_spectrum:
        # Compare a batch-average spectrum, as the reference does, over the batch only: the patches
        # stay separate on their own axis.
        pred_spectrum = pred_spectrum.mean(0, keepdim=True)
        target_spectrum = target_spectrum.mean(0, keepdim=True)

    spectrum_distance = pred_spectrum - target_spectrum
    # real**2 + imag**2, not abs()**2: the gradient of a complex modulus is undefined at 0.
    freq_distance = spectrum_distance.real**2 + spectrum_distance.imag**2

    if matrix is None:
        weight = freq_distance.sqrt() ** alpha

        if log_matrix:
            weight = (weight + 1.0).log()

        if batch_matrix:
            weight = weight / weight.max()
        else:
            weight = weight / weight.amax(dim=(-1, -2), keepdim=True)

        # An exactly matched spectrum normalizes to 0/0; its weight is 0, as in the reference.
        weight = weight.nan_to_num(0.0).clamp(0.0, 1.0).detach()
    else:
        weight = matrix.detach()

    loss = (weight * freq_distance).mean()

    return loss.to(output_dtype) if output_dtype in (torch.float16, torch.bfloat16) else loss


class FocalFrequencyLoss(nn.Module):
    r"""Weight the 2-D DFT distance between two images by its own magnitude (module form).

    ``criterion = FocalFrequencyLoss(loss_weight, alpha, patch_factor, ave_spectrum, log_matrix,
    batch_matrix)`` stores the options, and ``loss = criterion(pred, target)`` is
    ``focal_frequency_loss(pred, target, ...) * loss_weight``.

    Convention:
        See the Convention block of :func:`~kornia.losses.focal_frequency_loss`.

    Args:
        loss_weight: multiplier of the loss.
        alpha: exponent of the focal weight, as in :func:`~kornia.losses.focal_frequency_loss`.
        patch_factor: number of patches per side, as in :func:`~kornia.losses.focal_frequency_loss`.
        ave_spectrum: average the spectra over the batch before the distance.
        log_matrix: take the logarithm of the weight before it is normalized.
        batch_matrix: normalize the weight by its maximum over the whole batch.

    Shape:
        - pred: prediction ``(B, C, H, W)``.
        - target: target with the same shape as ``pred``.
        - Output: a scalar.

    See Also:
        - :func:`~kornia.losses.focal_frequency_loss`: the one-call form.
        - :class:`~kornia.losses.SSIMLoss`: a reconstruction criterion in the spatial domain.

    Example:
        >>> criterion = FocalFrequencyLoss(loss_weight=0.5)
        >>> pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        >>> target = torch.zeros_like(pred)
        >>> criterion(pred, target)
        tensor(3.3500)

    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def forward(self, pred: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor | None = None) -> torch.Tensor:
        """Weight the per-frequency distance between the spectra of ``pred`` and ``target``.

        Args:
            pred: Predicted tensor with shape ``(B, C, H, W)``.
            target: Target tensor with the same shape as ``pred``.
            matrix: optional predefined weight, as in :func:`~kornia.losses.focal_frequency_loss`.

        Returns:
            The scalar loss, multiplied by ``self.loss_weight``.
        """
        return self.loss_weight * focal_frequency_loss(
            pred=pred,
            target=target,
            alpha=self.alpha,
            patch_factor=self.patch_factor,
            ave_spectrum=self.ave_spectrum,
            log_matrix=self.log_matrix,
            batch_matrix=self.batch_matrix,
            matrix=matrix,
        )
