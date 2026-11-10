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

import functools

import pytest
import torch

import kornia

from testing.base import BaseTester

# The reference values of the MIT implementation this loss reimplements (EndlessSora/focal-frequency-loss,
# arXiv:2012.12821). They were produced by two independent reimplementations that agree to 1e-15: an
# op-by-op transcription of that package, and a NumPy implementation that builds the orthonormal 2-D DFT
# as an explicit sum, with no FFT call. Generated on CPU by
#   torch.manual_seed(0)
#   pred = torch.randn(2, 3, 8, 8, dtype=torch.float64)
#   target = torch.randn(2, 3, 8, 8, dtype=torch.float64)
#   reference(pred, target, **kwargs).item()
REFERENCE_CASES = [
    pytest.param({}, 1.2539364878316714, id="default"),
    pytest.param({"alpha": 2.0}, 0.8879437987567713, id="alpha_2"),
    pytest.param({"patch_factor": 2}, 1.494413958227291, id="patch_factor_2"),
    pytest.param({"patch_factor": 4, "alpha": 0.5}, 1.8760796334246617, id="patch_factor_4_alpha_half"),
    pytest.param({"log_matrix": True}, 1.4936484925279039, id="log_matrix"),
    pytest.param({"batch_matrix": True}, 1.001060636086098, id="batch_matrix"),
    pytest.param({"patch_factor": 2, "batch_matrix": True}, 1.040683594836554, id="patch_factor_2_batch_matrix"),
    pytest.param({"ave_spectrum": True}, 0.6488450628209851, id="ave_spectrum"),
    pytest.param(
        {"alpha": 2.0, "patch_factor": 2, "log_matrix": True, "batch_matrix": True, "ave_spectrum": True},
        0.4840748219858943,
        id="every_option",
    ),
]


class TestFocalFrequencyLoss(BaseTester):
    @pytest.mark.parametrize("patch_factor", [1, 2, 4])
    @pytest.mark.parametrize("shape", [(1, 2, 8, 8), (2, 3, 8, 12)])
    def test_smoke(self, device, dtype, shape, patch_factor):
        pred = torch.rand(shape, device=device, dtype=dtype)
        target = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.focal_frequency_loss(pred, target, patch_factor=patch_factor)

        assert actual.shape == ()
        assert actual.dtype == dtype

    def test_exception(self, device, dtype):
        img = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)

        # Check if both are tensors
        from kornia.core.exceptions import TypeCheckError

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.focal_frequency_loss(1.0, img)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            kornia.losses.focal_frequency_loss(img, 1.0)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        # Check the input is an image, not a channel-first volume
        from kornia.core.exceptions import ShapeError

        volume = torch.rand(3, 8, 8, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.losses.focal_frequency_loss(volume, volume)
        assert "Shape dimension mismatch: expected 4 dimensions, got 3" in str(errinfo.value)

        # Check if same shape
        with pytest.raises(ShapeError) as errinfo:
            kornia.losses.focal_frequency_loss(img, torch.rand(2, 3, 8, 9, device=device, dtype=dtype))
        assert "Shape mismatch" in str(errinfo.value)

        # Check if same device; a meta tensor stands for another device on a CPU-only runner
        from kornia.core.exceptions import DeviceError

        with pytest.raises(DeviceError) as errinfo:
            kornia.losses.focal_frequency_loss(img, torch.rand(2, 3, 8, 8, device="meta", dtype=dtype))
        assert "Device mismatch" in str(errinfo.value)

        # Check the patch factor divides the image
        from kornia.core.exceptions import BaseError

        with pytest.raises(BaseError) as errinfo:
            kornia.losses.focal_frequency_loss(img, img, patch_factor=3)
        assert "The patch factor must divide the image height and width" in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            kornia.losses.focal_frequency_loss(img, img, patch_factor=0)
        assert "The patch factor must be positive" in str(errinfo.value)

    @pytest.mark.parametrize("ave_spectrum", [False, True])
    @pytest.mark.parametrize("log_matrix", [False, True])
    @pytest.mark.parametrize("batch_matrix", [False, True])
    @pytest.mark.parametrize("patch_factor", [1, 2])
    def test_cardinality(self, device, dtype, patch_factor, batch_matrix, log_matrix, ave_spectrum):
        pred = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        target = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        options = {
            "patch_factor": patch_factor,
            "batch_matrix": batch_matrix,
            "log_matrix": log_matrix,
            "ave_spectrum": ave_spectrum,
        }

        actual = kornia.losses.focal_frequency_loss(pred, target, **options)
        assert actual.shape == ()
        assert actual.dtype == dtype

        actual = kornia.losses.FocalFrequencyLoss(**options)(pred, target)
        assert actual.shape == ()
        assert actual.dtype == dtype

    @pytest.mark.parametrize("shape", [(1, 2, 8, 8), (2, 3, 8, 12)])
    def test_perfect_prediction(self, device, dtype, shape):
        # An exactly matched spectrum normalizes its weight to 0/0, which the loss turns into the
        # zero weight, so identical inputs give exactly 0 and any mismatch gives more than 0.
        img = torch.rand(shape, device=device, dtype=dtype)

        actual = kornia.losses.focal_frequency_loss(img, img)
        self.assert_close(actual, torch.zeros((), device=device, dtype=dtype))

        actual = kornia.losses.focal_frequency_loss(img, torch.zeros_like(img))
        assert actual > torch.zeros((), device=device, dtype=dtype)

    @pytest.mark.parametrize("kwargs, expected", REFERENCE_CASES)
    def test_reference_values(self, device, dtype, kwargs, expected):
        # The literals are float64. The inputs are drawn in float64 on the CPU and cast, so a lower
        # precision only rounds them, which stays within the dtype tolerances of assert_close.
        torch.manual_seed(0)
        pred = torch.randn(2, 3, 8, 8, device="cpu", dtype=torch.float64).to(device=device, dtype=dtype)
        target = torch.randn(2, 3, 8, 8, device="cpu", dtype=torch.float64).to(device=device, dtype=dtype)

        self.assert_close(
            kornia.losses.focal_frequency_loss(pred, target, **kwargs),
            torch.tensor(expected, device=device, dtype=dtype),
        )

    def test_patch_per_pixel_is_mean_squared_error(self, device, dtype):
        # A patch of one pixel has a single frequency, whose normalized weight is 1, so the focal
        # weighting drops out entirely.
        pred = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        target = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)

        actual = kornia.losses.focal_frequency_loss(pred, target, patch_factor=4)

        self.assert_close(actual, torch.nn.functional.mse_loss(pred, target))

    def test_unit_weight_is_mean_squared_error(self, device, dtype):
        # A weight of ones leaves the plain mean spectrum distance, which the orthonormal transform
        # makes equal to the mean squared error.
        pred = torch.rand(2, 3, 8, 12, device=device, dtype=dtype)
        target = torch.rand(2, 3, 8, 12, device=device, dtype=dtype)
        matrix = torch.ones(2, 1, 3, 8, 12, device=device, dtype=dtype)

        actual = kornia.losses.focal_frequency_loss(pred, target, matrix=matrix)

        self.assert_close(actual, torch.nn.functional.mse_loss(pred, target))

    def test_gradcheck(self, device):
        # The dynamic weight is detached, so the recorded gradient is that of the distance under a
        # weight frozen at its current value, not the Jacobian of the full composite; a numerical
        # gradcheck of the composite therefore disagrees with it by design. A predefined matrix pins
        # the weight to a constant, which leaves exactly the differentiable core the backward pass
        # differentiates: the patching, the transform, real**2 + imag**2 and the weighted mean.
        pred = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64)
        target = torch.rand(2, 3, 4, 4, device=device, dtype=torch.float64)
        matrix = torch.rand(2, 4, 3, 2, 2, device=device, dtype=torch.float64)

        op = functools.partial(
            kornia.losses.focal_frequency_loss, patch_factor=2, alpha=2.0, log_matrix=True, matrix=matrix
        )
        self.gradcheck(op, (pred, target))

    def test_weight_detached_from_gradient(self, device, dtype):
        # Freezing the weight at the value it takes on the current inputs must reproduce the dynamic
        # gradient exactly, because that frozen loss is what the detached weight makes backprop
        # differentiate. The half-precision inputs are transformed in float32, as documented.
        pred = torch.rand(2, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
        target = torch.rand(2, 3, 4, 4, device=device, dtype=dtype)
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # one patch per image, so the spectrum keeps its own patch axis
        spectrum = torch.fft.fft2(pred.detach().to(compute_dtype).unsqueeze(1)) - torch.fft.fft2(
            target.to(compute_dtype).unsqueeze(1)
        )
        distance = spectrum.real**2 + spectrum.imag**2
        matrix = (distance.sqrt() / distance.sqrt().amax(dim=(-1, -2), keepdim=True)).clamp(0.0, 1.0)

        loss = kornia.losses.focal_frequency_loss(pred, target)
        grad_dynamic = torch.autograd.grad(loss, pred)[0]

        pred_frozen = pred.detach().clone().requires_grad_(True)
        loss_frozen = kornia.losses.focal_frequency_loss(pred_frozen, target, matrix=matrix)
        grad_frozen = torch.autograd.grad(loss_frozen, pred_frozen)[0]

        self.assert_close(grad_dynamic, grad_frozen)

    def test_dynamo(self, device, dtype, torch_optimizer):
        img1 = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)

        op = kornia.losses.focal_frequency_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(
            op(img1, img2, alpha=2.0, patch_factor=2), op_optimized(img1, img2, alpha=2.0, patch_factor=2)
        )

        op_module = kornia.losses.FocalFrequencyLoss(alpha=2.0, patch_factor=2)
        op_module_optimized = torch_optimizer(op_module)

        self.assert_close(op_module(img1, img2), op_module_optimized(img1, img2))

    def test_module(self, device, dtype):
        img1 = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        img2 = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)

        op = kornia.losses.focal_frequency_loss
        op_module = kornia.losses.FocalFrequencyLoss()

        self.assert_close(op(img1, img2), op_module(img1, img2))

        # The functional returns the unweighted mean; only the module carries loss_weight.
        weighted = kornia.losses.FocalFrequencyLoss(loss_weight=2.0, alpha=2.0, patch_factor=2)(img1, img2)
        self.assert_close(weighted, 2.0 * op(img1, img2, alpha=2.0, patch_factor=2))

    def test_half_precision_computed_in_float32(self, device, dtype):
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("only half-precision inputs are evaluated in another dtype")
        pred = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        target = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)

        out = kornia.losses.focal_frequency_loss(pred, target, patch_factor=2, alpha=2.0, log_matrix=True)
        in_float32 = kornia.losses.focal_frequency_loss(
            pred.float(), target.float(), patch_factor=2, alpha=2.0, log_matrix=True
        ).to(dtype)

        assert out.dtype == dtype
        self.assert_close(out, in_float32, rtol=0, atol=0)
