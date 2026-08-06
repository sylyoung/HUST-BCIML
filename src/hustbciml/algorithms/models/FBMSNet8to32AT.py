# ===========================================================================
# Implementation of: FBMSNet 8-32 Hz architecture transfer

# Credit chain († = co-first authors; every node except the integrator carries its GitHub link):
#   Original authors:    Ke Liu, Mingzhao Yang, Zhuliang Yu, Guoyin Wang, Wei Wu (2023) — "FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery Decoding", IEEE Trans. Biomed. Eng.
#                        Original code: https://github.com/Want2Vanish/FBMSNet (pinned 1c6b659)
#   Implementation:      Ke Liu et al. — Want2Vanish/FBMSNet (https://github.com/Want2Vanish/FBMSNet) (official)
#   Current code:        Siyang Li — ported from pinned Want2Vanish/FBMSNet (https://github.com/Want2Vanish/FBMSNet) commit (8-32 Hz architecture transfer)
#   Integrated by:       Siyang Li <lsyyoungll@gmail.com> — HUST-BCIML

# References (IEEE BibTeX):
#   @Article{Liu2023,
#     author  = {Liu, Ke and Yang, Mingzhao and Yu, Zhuliang and Wang, Guoyin and Wu, Wei},
#     journal = {IEEE Transactions on Biomedical Engineering},
#     title   = {{FBMSNet}: A Filter-Bank Multi-Scale Convolutional Neural Network for {EEG}-Based Motor Imagery Decoding},
#     year    = {2023},
#     number  = {2},
#     pages   = {436-445},
#     volume  = {70},
#     doi     = {10.1109/TBME.2022.3193277},
#   }
# ===========================================================================
"""Compliant 8–32 Hz adaptation of the released FBMSNet feature extractor.

The original nine-view 4–40 Hz bank cannot be used in this benchmark because
4–8 Hz motor-imagery processing is prohibited. This explicitly named transfer
uses six causal 4-Hz views within 8–32 Hz. The released Chebyshev-II design and
``lfilter`` zero-state behavior are preserved by finite-epoch FFT convolution.
The common benchmark's broad 8–32 Hz preprocessing, shared linear head, and
cross-entropy objective remain in force, so this is not a paper-protocol
reproduction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.core.stages import Backbone
from hustbciml.utils.shapes import probe


class _Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = float(max_norm)
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.copy_(
                torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
            )
        return super().forward(x)


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _LogVarLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=3, keepdim=True)
        return torch.log(torch.clamp(variance, 1e-6, 1e6))


def _split_channels(num_channels: int, num_groups: int) -> list[int]:
    split = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


class _MixedConv2d(nn.ModuleDict):
    """Released mixed-width temporal convolution with TensorFlow SAME padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.in_splits = _split_channels(in_channels, len(self.kernel_sizes))
        out_splits = _split_channels(out_channels, len(self.kernel_sizes))
        for index, (kernel, in_count, out_count) in enumerate(
            zip(self.kernel_sizes, self.in_splits, out_splits)
        ):
            self.add_module(
                str(index),
                nn.Conv2d(
                    in_count,
                    out_count,
                    kernel,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

    @staticmethod
    def _same_pad(x: torch.Tensor, kernel_size) -> torch.Tensor:
        pad_width = int(kernel_size[-1]) - 1
        return F.pad(x, [pad_width // 2, pad_width - pad_width // 2, 0, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = torch.split(x, self.in_splits, dim=1)
        outputs = [
            convolution(self._same_pad(inputs[index], self.kernel_sizes[index]))
            for index, convolution in enumerate(self.values())
        ]
        return torch.cat(outputs, dim=1)


class FBMSNet8to32AT(Backbone):
    """Six-view causal FBMSNet transfer for ``(B, 1, C, T)`` input."""

    task_name = "classification"
    bands = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 28.0),
        (28.0, 32.0),
    )

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes: int,
        sfreq: float,
        num_feat: int = 36,
        dilatability: int = 8,
        stride_factor: int = 4,
        **_,
    ):
        super().__init__()
        if n_classes != 2:
            raise ValueError(
                f"FBMSNet-8-32-AT is reportable only for the two-class MI "
                f"protocol, got n_classes={n_classes}"
            )
        if n_times < stride_factor:
            raise ValueError(
                f"FBMSNet-8-32-AT needs at least {stride_factor} samples, got {n_times}"
            )

        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.n_bands = len(self.bands)
        self.num_feat = int(num_feat)
        self.dilatability = int(dilatability)
        self.stride_factor = int(stride_factor)
        self.fft_length = 1 << (2 * self.n_times - 1).bit_length()

        impulse_responses, coefficients, orders = self._design_filter_bank()
        self.filter_coefficients = coefficients
        self.filter_orders = orders
        self.register_buffer(
            "filter_impulse_responses",
            torch.from_numpy(impulse_responses.astype(np.float32)),
        )
        frequency_response = torch.fft.rfft(
            self.filter_impulse_responses,
            n=self.fft_length,
            dim=-1,
        )
        self.register_buffer(
            "filter_frequency_response",
            frequency_response,
        )

        self.mix_conv = nn.Sequential(
            _MixedConv2d(
                self.n_bands,
                self.num_feat,
                kernel_sizes=[(1, 15), (1, 31), (1, 63), (1, 125)],
            ),
            nn.BatchNorm2d(self.num_feat),
        )
        self.scb = nn.Sequential(
            _Conv2dWithConstraint(
                self.num_feat,
                self.num_feat * self.dilatability,
                (self.n_chans, 1),
                groups=self.num_feat,
                max_norm=2.0,
                padding=0,
            ),
            nn.BatchNorm2d(self.num_feat * self.dilatability),
            _Swish(),
        )
        self.temporal_layer = _LogVarLayer()

        with probe(self):
            features = self.forward_features(
                torch.zeros(1, 1, self.n_chans, self.n_times)
            )
        self.out_features = int(features.shape[1])

    def _design_filter_bank(self):
        from scipy.signal import cheb2ord, cheby2, lfilter

        nyquist = self.sfreq / 2.0
        allowance = 2.0
        pass_attenuation = 3.0
        stop_attenuation = 30.0
        impulse = np.zeros(self.n_times, dtype=np.float64)
        impulse[0] = 1.0
        responses = []
        coefficients = []
        orders = []

        for low, high in self.bands:
            pass_band = np.asarray([low, high], dtype=np.float64) / nyquist
            stop_band = np.asarray(
                [low - allowance, high + allowance], dtype=np.float64
            ) / nyquist
            if not (0 < stop_band[0] < pass_band[0] < pass_band[1] < stop_band[1] < 1):
                raise ValueError(
                    f"FBMSNet-8-32-AT cannot realize {low:g}-{high:g} Hz at "
                    f"sfreq={self.sfreq:g} Hz"
                )
            try:
                order, natural_frequency = cheb2ord(
                    pass_band,
                    stop_band,
                    pass_attenuation,
                    stop_attenuation,
                )
                # The pinned release computes ``natural_frequency`` but passes
                # the requested stop-band edges to cheby2 (transforms.py:144-145).
                # Preserve that released design rather than silently replacing it
                # with SciPy's natural-frequency design.
                _ = natural_frequency
                numerator, denominator = cheby2(
                    order,
                    stop_attenuation,
                    stop_band,
                    btype="bandpass",
                )
            except Exception as exc:
                raise ValueError(
                    f"FBMSNet-8-32-AT filter design failed for {low:g}-{high:g} Hz "
                    f"at sfreq={self.sfreq:g} Hz"
                ) from exc
            responses.append(lfilter(numerator, denominator, impulse))
            coefficients.append({
                "band": [low, high],
                "numerator": numerator.tolist(),
                "denominator": denominator.tolist(),
            })
            orders.append(int(order))

        return np.stack(responses), coefficients, orders

    def causal_filter_bank(self, x: torch.Tensor) -> torch.Tensor:
        """Apply zero-state causal IIR views through equivalent FFT convolution."""
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] != self.n_times:
            raise ValueError(
                f"FBMSNet-8-32-AT expects (B, 1, C, {self.n_times}), got "
                f"{tuple(x.shape)}"
            )
        signal = x[:, 0]  # (B, C, T)
        spectrum = torch.fft.rfft(signal, n=self.fft_length, dim=-1)
        response = self.filter_frequency_response.to(dtype=spectrum.dtype)
        filtered = torch.fft.irfft(
            spectrum[:, None, :, :] * response[None, :, None, :],
            n=self.fft_length,
            dim=-1,
        )
        return filtered[..., : self.n_times]  # (B, six bands, C, T)

    def split_temporal_segments(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split contiguously into near-equal segments without dropping samples."""
        segments = torch.tensor_split(x, self.stride_factor, dim=3)
        if sum(segment.shape[3] for segment in segments) != x.shape[3]:
            raise RuntimeError("temporal segmentation dropped samples")
        return tuple(segments)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.causal_filter_bank(x)
        x = self.mix_conv(x)
        x = self.scb(x)
        log_variances = [
            self.temporal_layer(segment)
            for segment in self.split_temporal_segments(x)
        ]
        x = torch.cat(log_variances, dim=2)
        return x.flatten(1)
