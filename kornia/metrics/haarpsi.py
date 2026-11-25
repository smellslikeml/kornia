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

from typing import List

import torch
import torch.nn.functional as F

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


def _rgb_to_yiq(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YIQ colour space, keeping the shape ``(B, 3, H, W)``."""
    rgb_to_yiq: torch.Tensor = torch.tensor(
        [[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]],
        device=image.device,
        dtype=image.dtype,
    )
    return (image.permute(0, 2, 3, 1) @ rgb_to_yiq.t()).permute(0, 3, 1, 2)


def _haar_filter(kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a Haar wavelet low/high kernel of shape ``(1, kernel_size, kernel_size)``."""
    kernel = torch.ones([kernel_size, kernel_size], device=device, dtype=dtype) / kernel_size
    kernel[kernel_size // 2 :] = -kernel[kernel_size // 2 :]
    return kernel.unsqueeze(0)


def _similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float) -> torch.Tensor:
    """Compute the similarity map of two absolute coefficients arrays."""
    return (2 * map_x * map_y + constant) / (map_x**2 + map_y**2 + constant)


def haar_psi(
    img1: torch.Tensor,
    img2: torch.Tensor,
    reduction: str = "mean",
    data_range: float = 1.0,
    scales: int = 3,
    subsample: bool = True,
    c: float = 30.0,
    alpha: float = 4.2,
) -> torch.Tensor:
    r"""Compute the Haar Wavelet-Based Perceptual Similarity Index (HaarPSI) between two images.

    ``score = haar_psi(img1, img2)`` takes two ``(B, C, H, W)`` images and returns one perceptual similarity in
    :math:`[0, 1]` per sample, where a higher value means more similar; 1 holds for identical images. The metric was
    introduced in :cite:`reisenhofer2018haarpsi`; the reference port defaults reproduce it.

    .. math::

      \mathrm{HaarPSI}(x, y) = \left(\frac{\mathcal{L}\left(\sum_{\omega, i}
      w_\omega \,\sigma_\alpha\left(\mathrm{S}_\omega^i(x, y)\right)\right)}{\alpha}\right)^2

    where :math:`\mathcal{L}` is the logistic function, :math:`\sigma_\alpha(\cdot)` the sigmoid scaled by the slope
    parameter :math:`\alpha`, :math:`\mathrm{S}` the per-pixel local similarity of two Haar coefficient magnitudes,
    and :math:`w` the magnitude of the coarsest-scale coefficients, which also weights the chroma similarity of a
    colour pair.

    Convention:
        - The score is in :math:`[0, 1]`, higher is more similar, and an identical pair scores 1. It is the inverse
          orientation of :func:`~kornia.metrics.ssim`, whose map is averaged for a similarity too.
        - The images are rescaled to ``[0, 255]``; the scale constants are tuned for that range, so ``data_range``
          must match the actual range of the inputs. Colour images are converted to YIQ, and the chroma channels
          contribute to the score.
        - ``c`` (default ``30.0``) stabilises the similarity ratio and ``alpha`` (default ``4.2``) is the sigmoid
          slope; the reference implementation defaults are reproduced exactly.

    Args:
        img1: the first image with shape ``(B, C, H, W)`` and ``C`` in ``{1, 3}``; RGB order for ``C == 3``.
        img2: the second image with the shape of ``img1``.
        reduction: ``'mean'`` (default) averages the score over the batch, ``'sum'`` adds it, ``'none'`` returns a
            per-sample vector of shape ``(B,)``.
        data_range: data range of the images, a Python float or int.
        scales: number of Haar wavelet scales, a positive integer.
        subsample: downsample the images by a factor of two first, as the reference does before the wavelet
            decomposition.
        c: similarity stabilising constant.
        alpha: slope of the logistic function applied before the pooling.

    Returns:
        the similarity score, a scalar for ``'mean'`` and ``'sum'`` and a ``(B,)`` vector for ``'none'``.

    Note:
        Integer images and half-precision inputs are evaluated like :func:`~kornia.metrics.ssim`: integers count as
        float32, and a mixed pair is computed in the promoted dtype of the inputs.

    See Also:
        - :func:`~kornia.metrics.ssim`: a per-pixel map instead of one pooled score.
        - :func:`~kornia.metrics.psnr`: a pixel-error measure of the same pair, as one score in dB.

    Examples:
        >>> input1 = torch.rand(1, 1, 32, 32)
        >>> input2 = torch.rand(1, 1, 32, 32)
        >>> score = haar_psi(input1, input2)  # scalar in [0, 1]

    """
    KORNIA_CHECK_IS_TENSOR(img1)
    KORNIA_CHECK_IS_TENSOR(img2)
    KORNIA_CHECK_SHAPE(img1, ["B", "C", "H", "W"])
    KORNIA_CHECK(img1.shape == img2.shape, f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")
    KORNIA_CHECK(
        img1.shape[1] in (1, 3),
        f"Expected grayscale (1-channel) or RGB (3-channel) images. Got: {img1.shape[1]} channels",
    )
    KORNIA_CHECK(
        isinstance(data_range, (float, int)) and data_range > 0, f"data_range must be positive. Got: {data_range}"
    )

    # Preserve fractional weights and avoid integer rescaling; compute in the promoted dtype of the pair.
    if not img1.is_floating_point() and not img1.is_complex():
        img1 = img1.to(torch.float32)
    if not img2.is_floating_point() and not img2.is_complex():
        img2 = img2.to(torch.float32)
    compute_dtype = torch.promote_types(img1.dtype, img2.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    img1, img2 = img1.to(compute_dtype), img2.to(compute_dtype)

    x: torch.Tensor = img1 / float(data_range) * 255.0
    y: torch.Tensor = img2 / float(data_range) * 255.0

    if x.shape[1] == 3:
        x_yiq = _rgb_to_yiq(x)
        y_yiq = _rgb_to_yiq(y)
    else:
        x_yiq = x
        y_yiq = y

    if subsample:
        x_yiq = F.avg_pool2d(F.pad(x_yiq, [0, x_yiq.shape[3] % 2, 0, x_yiq.shape[2] % 2]), kernel_size=2, stride=2)
        y_yiq = F.avg_pool2d(F.pad(y_yiq, [0, y_yiq.shape[3] % 2, 0, y_yiq.shape[2] % 2]), kernel_size=2, stride=2)

    coefficients_x: List[torch.Tensor] = []
    coefficients_y: List[torch.Tensor] = []
    for scale in range(scales):
        kernel_size = 2 ** (scale + 1)
        kernel = _haar_filter(kernel_size, device=x.device, dtype=x.dtype)
        kernels = torch.stack([kernel, kernel.transpose(-1, -2)], dim=0)
        # Asymmetric even-kernel padding matches the MATLAB conv2(..., 'same') convention of the reference.
        padding = [kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2]
        coefficients_x.append(F.conv2d(F.pad(x_yiq[:, :1], padding), kernels, stride=1))
        coefficients_y.append(F.conv2d(F.pad(y_yiq[:, :1], padding), kernels, stride=1))

    coefficients_x = torch.cat(coefficients_x, dim=1)
    coefficients_y = torch.cat(coefficients_y, dim=1)

    # Weight every pixel by the magnitude of the coarsest-scale coefficients (the last two channels at the default
    # `scales=3`, matching the reference implementation's slice).
    weights: torch.Tensor = torch.max(torch.abs(coefficients_x[:, 4:]), torch.abs(coefficients_y[:, 4:]))

    sim_map: List[torch.Tensor] = []
    for orientation in (0, 1):
        mx = torch.abs(coefficients_x[:, (orientation, orientation + 2)])
        my = torch.abs(coefficients_y[:, (orientation, orientation + 2)])
        sim_map.append(_similarity_map(mx, my, c).sum(dim=1, keepdim=True) / 2)

    if x.shape[1] == 3:
        cx_iq = torch.abs(F.avg_pool2d(F.pad(x_yiq[:, 1:], [0, 1, 0, 1]), kernel_size=2, stride=1))
        cy_iq = torch.abs(F.avg_pool2d(F.pad(y_yiq[:, 1:], [0, 1, 0, 1]), kernel_size=2, stride=1))
        weights = torch.cat([weights, weights.mean(dim=1, keepdim=True)], dim=1)
        sim_map.append(_similarity_map(cx_iq, cy_iq, c).sum(dim=1, keepdim=True) / 2)

    sim_map = torch.cat(sim_map, dim=1)

    eps = torch.finfo(x.dtype).eps
    score = (torch.sigmoid(sim_map * alpha) * weights).sum(dim=(1, 2, 3))
    score = (score + eps) / (weights.sum(dim=(1, 2, 3)) + eps)
    score = (torch.log(score / (1 - score)) / alpha) ** 2

    if reduction == "mean":
        score = torch.mean(score)
    elif reduction == "sum":
        score = torch.sum(score)
    elif reduction != "none":
        raise NotImplementedError("Invalid reduction option.")

    return score
