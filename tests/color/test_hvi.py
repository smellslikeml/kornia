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
from torch.autograd import gradcheck

import kornia

from testing.base import BaseTester


class TestRgbToHvi(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_hvi(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_hvi(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_hvi([0.0])

        img = torch.ones(1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.rgb_to_hvi(img)

        img = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.rgb_to_hvi(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.5, 0.0, 1.0, 0.2600],
                ],
                [
                    [0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.5, 0.0, 1.0, 0.4173],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.5, 0.0, 1.0, 0.0259],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # The primaries, secondaries, gray, black, white and one mixed pixel, at density_k = 0.2.
        # Expected generated, independently of kornia, in float64 from the reference transform of
        # Fediory/HVI-CIDNet (MIT), net/HVI_transform.py, RGB_HVI.HVIT, with eps = 1e-8:
        #   d = max - min + eps; hue = (4 + (r-g)/d)[b max] / (2 + (b-r)/d)[g max] / ((g-b)/d)[r max],
        #   masks assigned in blue, green, red order (red wins ties, as in rgb_to_hsv), 0 where min == max,
        #   then hue /= 6; saturation = (max-min)/(max+eps), 0 at black; cs = (sin(max*pi/2) + eps)**0.2
        #   out = [cs*saturation*cos(2*pi*hue), cs*saturation*sin(2*pi*hue), max]
        expected = torch.tensor(
            [
                [
                    [1.0000000, -0.5000000, -0.5000000, 0.5000000, -1.0000000],
                    [0.5000000, 0.0000000, 0.0000000, 0.0000000, 0.0871262],
                ],
                [
                    [0.0000000, 0.8660254, -0.8660254, 0.8660254, 0.0000000],
                    [-0.8660254, 0.0000000, 0.0000000, 0.0000000, 0.8450418],
                ],
                [
                    [1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000],
                    [1.0000000, 0.5000000, 0.0000000, 1.0000000, 0.4173000],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # float16 rounds the hue in radians before the trigonometry, moving the chroma by more than the
        # default tolerance: compare at half the decimal places.
        self.assert_close(kornia.color.rgb_to_hvi(data), expected, low_tolerance=True)

    def test_roundtrip(self, device, dtype):
        # The color sensitivity cancels between the two calls. The eps floors keep hue and saturation
        # defined where the chroma vanishes, so grayscale pixels return with a saturation of sqrt(eps) and
        # their RGB moves by up to V * sqrt(eps), about 5e-5; float16 additionally rounds the hue twice,
        # once per trigonometry: compare at half the decimal places.
        random = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        fixed = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.5]],
                [[0.0, 1.0, 0.0, 0.5]],
                [[0.0, 0.0, 1.0, 0.5]],
            ],
            device=device,
            dtype=dtype,
        )
        for image in (random, fixed):
            roundtrip = kornia.color.hvi_to_rgb(kornia.color.rgb_to_hvi(image))
            self.assert_close(roundtrip, image, low_tolerance=True)

    @pytest.mark.parametrize("density_k", [0.2, 1.0])
    def test_roundtrip_density_k(self, device, dtype, density_k):
        # The encode and the decode must agree on the exponent: the color sensitivity still cancels.
        data = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        roundtrip = kornia.color.hvi_to_rgb(kornia.color.rgb_to_hvi(data, density_k), density_k)
        self.assert_close(roundtrip, data, low_tolerance=True)

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_hvi, (img,), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_hvi
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        data = torch.tensor([[[1e-4]], [[0.0]], [[0.0]]], device=device, dtype=dtype)
        ops = kornia.color.RgbToHvi().to(device, dtype)
        fcn = kornia.color.rgb_to_hvi
        self.assert_close(ops(data), fcn(data))

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_hvi
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))


class TestHviToRgb(BaseTester):
    def test_smoke(self, device, dtype):
        C, H, W = 3, 4, 5
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.hvi_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize("shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.hvi_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        with pytest.raises(TypeError):
            assert kornia.color.hvi_to_rgb([0.0])

        img = torch.ones(1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.hvi_to_rgb(img)

        img = torch.ones(2, 1, 1, device=device, dtype=dtype)
        with pytest.raises(ValueError):
            assert kornia.color.hvi_to_rgb(img)

    def test_unit(self, device, dtype):
        data = torch.tensor(
            [
                [
                    [1.0000000, -0.5000000, -0.5000000, 0.0000000, 0.2500000],
                    [-0.3000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
                [
                    [0.0000000, 0.8660254, -0.8660254, 0.0000000, -0.4000000],
                    [-0.4500000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
                [
                    [1.0000000, 1.0000000, 1.0000000, 0.5000000, 0.8000000],
                    [0.6000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # The forward HVI of the three primaries, a 0.5 gray, and two mixed triples of which the last has
        # negative chroma on both axes, so the decoded hue crosses the negative-radian wrap.
        # Expected generated, independently of kornia, in float64 from the reference transform of
        # Fediory/HVI-CIDNet (MIT), net/HVI_transform.py, RGB_HVI.PHVIT, with eps = 1e-8 (its H/V and
        # saturation clamps are inactive inside the color-sensitivity disk these pixels occupy):
        #   cs = (sin(I*pi/2) + eps)**0.2; H = H/(cs+eps); V = V/(cs+eps)
        #   hue = (atan2(V+eps, H+eps)/2pi) % 1; saturation = sqrt(H**2 + V**2 + eps)
        #   standard HSV-to-RGB reconstruction of (hue, saturation, I)
        expected = torch.tensor(
            [
                [
                    [1.0000000, 0.0000000, 0.0000000, 0.5000000, 0.8000000],
                    [0.2614501, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
                [
                    [0.0000000, 1.0000000, 0.0000000, 0.4999875, 0.4188344],
                    [0.2822713, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
                [
                    [0.0000000, 0.0000000, 1.0000000, 0.4999500, 0.7872603],
                    [0.6000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        # float16 rounds the decoded hue, moving the RGB by more than the default tolerance: compare at
        # half the decimal places.
        self.assert_close(kornia.color.hvi_to_rgb(data), expected, low_tolerance=True)

    def test_gradcheck(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.hvi_to_rgb, (img,), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.hvi_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        data = torch.tensor([[[0.0]], [[0.0]], [[0.5]]], device=device, dtype=dtype)
        ops = kornia.color.HviToRgb().to(device, dtype)
        fcn = kornia.color.hvi_to_rgb
        self.assert_close(ops(data), fcn(data))

    def test_dynamo(self, device, dtype, torch_optimizer):
        B, C, H, W = 2, 3, 4, 4
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.hvi_to_rgb
        op_optimized = torch_optimizer(op)
        self.assert_close(op(img), op_optimized(img))
