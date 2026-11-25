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

import pytest
import torch

import kornia
from kornia.metrics import haar_psi

from testing.base import BaseTester


def _noisy_pair(channels: int, seed: int = 7, noise: float = 0.1):
    """A fixed image pair used by the hardcoded-value and parity tests; generated in float32 and cast."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(1, channels, 16, 16, generator=generator)
    y = (x + noise * torch.randn(1, channels, 16, 16, generator=generator)).clamp(0, 1)
    return x, y


class TestHaarPsi(BaseTester):
    @pytest.mark.parametrize("channels", [1, 3])
    def test_identity(self, device, dtype, channels):
        # Identical images score 1: every similarity map is 1, so the pooled logit is exactly `alpha`.
        x = torch.rand(2, channels, 32, 33, device=device, dtype=dtype)
        self.assert_close(
            haar_psi(x, x, reduction="none"), torch.ones(2, device=device, dtype=dtype), atol=1e-5, rtol=0
        )
        self.assert_close(haar_psi(x, x), torch.ones_like(haar_psi(x, x)), atol=1e-5, rtol=0)

    @pytest.mark.parametrize("reduction,shape", [("none", (2,)), ("mean", ()), ("sum", ())])
    def test_reduction_shape(self, device, dtype, reduction, shape):
        x = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        score = haar_psi(x, y, reduction=reduction)
        assert score.shape == shape

    def test_range(self, device, dtype):
        x = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        score = haar_psi(x, y, reduction="none")
        assert bool((score > 0).all()) and bool((score <= 1).all())

    def test_monotonicity(self, device, dtype):
        # A more distorted image scores strictly lower than a less distorted one.
        x = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        generator = torch.Generator().manual_seed(0)
        less = (x + 0.1 * torch.randn(x.shape, generator=generator, device=device, dtype=dtype)).clamp(0, 1)
        more = (x + 0.3 * torch.randn(x.shape, generator=generator, device=device, dtype=dtype)).clamp(0, 1)
        assert bool((haar_psi(x, less, reduction="none") > haar_psi(x, more, reduction="none")).all())

    def test_gradient_flow(self, device, dtype):
        x = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = y.clone().requires_grad_(True)
        haar_psi(x, y).sum().backward()
        assert y.grad is not None
        assert bool(torch.isfinite(y.grad).all())
        assert float(y.grad.abs().sum()) > 0

    def test_gradcheck(self, device):
        x, y = _noisy_pair(1)
        self.gradcheck(haar_psi, (x.to(device), y.to(device)))

    def test_hardcoded_value(self, device, dtype):
        # Expected literals generated on CPU float32 via
        #   x, y = _noisy_pair(1); haar_psi(x, y, reduction='none')   -> tensor([0.7874])
        #   x, y = _noisy_pair(3); haar_psi(x, y, reduction='none')   -> tensor([0.8149])
        expected = {1: 0.7874, 3: 0.8149}
        for channels, value in expected.items():
            x, y = _noisy_pair(channels)
            self.assert_close(
                haar_psi(x.to(device, dtype), y.to(device, dtype), reduction="none"),
                torch.tensor([value], device=device, dtype=dtype),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_data_range(self, device, dtype):
        # The score is invariant under a common rescaling of the images and their data range.
        x = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        unit = haar_psi(x, y, data_range=1.0, reduction="none")
        bounded = haar_psi(x * 255, y * 255, data_range=255.0, reduction="none")
        self.assert_close(unit, bounded, rtol=0, atol=1e-4)

    def test_loss(self, device, dtype):
        x = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        criterion = kornia.losses.HaarPSILoss(reduction="none")
        self.assert_close(criterion(x, y), 1 - haar_psi(x, y, reduction="none"), rtol=0, atol=0)
        self.assert_close(
            kornia.losses.HaarPSILoss()(x, y),
            torch.tensor(1.0, device=device, dtype=dtype) - haar_psi(x, y),
            rtol=0,
            atol=0,
        )

    def test_loss_gradient_flow(self, device, dtype):
        x = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = torch.rand(2, 1, 32, 32, device=device, dtype=dtype)
        y = y.clone().requires_grad_(True)
        kornia.losses.HaarPSILoss()(x, y).backward()
        assert y.grad is not None
        assert bool(torch.isfinite(y.grad).all())

    def test_invalid_channels(self, device, dtype):
        x = torch.rand(2, 2, 32, 32, device=device, dtype=dtype)
        with pytest.raises(Exception):
            haar_psi(x, x)

    @pytest.mark.skipif(torch.cuda.is_available(), reason="piq parity is checked on CPU")
    def test_parity_with_piq(self):
        pytest.importorskip("piq")
        import piq

        for channels in (1, 3):
            x, y = _noisy_pair(channels)
            torch.testing.assert_close(
                haar_psi(x, y, reduction="none"),
                piq.haarpsi(x, y, reduction="none"),
                atol=1e-4,
                rtol=0,
            )
