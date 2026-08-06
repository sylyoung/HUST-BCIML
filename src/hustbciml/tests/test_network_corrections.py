"""Deterministic guards for the corrected Network-table identities."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.signal import cheb2ord, cheby2, lfilter

from hustbciml.algorithms.models.ADFCNNTransposeAT import (
    ADFCNNTransposeAT,
    _transpose_attention_output,
)
from hustbciml.algorithms.models.Deep4NetAT import Deep4NetAT
from hustbciml.algorithms.models.EEGWaveNetReleaseAT import EEGWaveNetReleaseAT
from hustbciml.algorithms.models.FBMSNet8to32AT import FBMSNet8to32AT
from hustbciml.algorithms.models.ShallowFBCSPNetAT import ShallowFBCSPNetAT
from hustbciml.algorithms.network_methods import NETWORK_METHODS
from hustbciml.core import registry
from hustbciml.core.pipeline import PipelineModel
from hustbciml.algorithms.heads.Linear import Linear


CORRECTED = {
    "Deep4NetAT",
    "ShallowFBCSPNetAT",
    "ADFCNNTransposeAT",
    "EEGWaveNetReleaseAT",
    "FBMSNet8to32AT",
}
LEGACY = {"DeepConvNet", "ShallowConvNet", "ADFCNN", "EEGWaveNet", "FBMSNet"}


def test_reportable_network_manifest_has_only_corrected_identities():
    assert len(NETWORK_METHODS) == 18
    backbones = {method.backbone for method in NETWORK_METHODS}
    assert CORRECTED <= backbones
    assert LEGACY.isdisjoint(backbones)
    assert "MVCNet" not in backbones
    assert all(method.n_classes == 2 for method in NETWORK_METHODS)
    assert all(method.input_view == "moabb_8_32_hz" for method in NETWORK_METHODS)


def test_registry_exposes_corrected_modules_not_legacy_modules():
    available = set(registry.available("models"))
    assert CORRECTED <= available
    assert LEGACY.isdisjoint(available)


def test_deep4net_layer_signature_and_reference_initialization(monkeypatch):
    calls = []
    original = nn.init.xavier_uniform_

    def capture(tensor, gain=1.0):
        calls.append(tuple(tensor.shape))
        return original(tensor, gain=gain)

    monkeypatch.setattr(nn.init, "xavier_uniform_", capture)
    model = Deep4NetAT(22, 1001, 2, 250.0)
    assert model.features.conv_time.kernel_size == (1, 10)
    assert model.features.conv_spat.kernel_size == (22, 1)
    assert model.features.pool_1.kernel_size == (1, 3)
    assert model.features.pool_1.stride == (1, 3)
    assert model.features.conv_time.bias is not None
    assert model.features.conv_spat.bias is None
    assert model.features.conv_2.bias is None
    assert model.features.conv_3.bias is None
    assert model.features.conv_4.bias is None
    assert len([layer for layer in model.modules() if isinstance(layer, nn.Dropout)]) == 3
    assert len(calls) == 5
    assert model.out_features == 1400


def test_shallow_fbcsp_layer_signature_and_reference_initialization(monkeypatch):
    calls = []
    original = nn.init.xavier_uniform_

    def capture(tensor, gain=1.0):
        calls.append(tuple(tensor.shape))
        return original(tensor, gain=gain)

    monkeypatch.setattr(nn.init, "xavier_uniform_", capture)
    model = ShallowFBCSPNetAT(22, 1001, 2, 250.0)
    assert model.features.conv_time.kernel_size == (1, 25)
    assert model.features.conv_spat.kernel_size == (22, 1)
    assert model.features.pool.kernel_size == (1, 75)
    assert model.features.pool.stride == (1, 15)
    assert model.features.conv_time.bias is not None
    assert model.features.conv_spat.bias is None
    assert len(calls) == 2
    assert model.out_features == 2440


def test_adfcnn_uses_a_real_attention_transpose():
    source = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
    corrected = _transpose_attention_output(source)
    assert corrected.shape == (2, 5, 3)
    assert torch.equal(corrected, source.permute(0, 2, 1))
    assert not torch.equal(corrected, source.reshape(2, 5, 3))
    with pytest.raises(ValueError, match="two-class"):
        ADFCNNTransposeAT(22, 1001, 4, 250.0)


def test_eegwavenet_release_minimum_length_is_explicit():
    with pytest.raises(ValueError, match="at least 448"):
        EEGWaveNetReleaseAT(22, 447, 2, 250.0)
    assert EEGWaveNetReleaseAT(22, 448, 2, 250.0).out_features == 32


@pytest.mark.parametrize(
    "n_times,sfreq,expected_segments",
    [(1001, 250.0, [251, 250, 250, 250]), (2561, 512.0, [641, 640, 640, 640])],
)
def test_fbmsnet_uses_only_allowed_bands_and_keeps_every_sample(
    n_times, sfreq, expected_segments
):
    model = FBMSNet8to32AT(3, n_times, 2, sfreq)
    assert model.bands == (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 28.0),
        (28.0, 32.0),
    )
    assert min(low for low, _ in model.bands) >= 8.0
    dummy = torch.zeros(1, 2, 1, n_times)
    segments = model.split_temporal_segments(dummy)
    assert [segment.shape[-1] for segment in segments] == expected_segments
    assert sum(segment.shape[-1] for segment in segments) == n_times
    assert model.out_features == 1152


def test_fbmsnet_filter_coefficients_match_the_pinned_release_design():
    model = FBMSNet8to32AT(3, 1001, 2, 250.0)
    nyquist = model.sfreq / 2
    for (low, high), stored in zip(model.bands, model.filter_coefficients):
        pass_band = np.asarray([low, high]) / nyquist
        stop_band = np.asarray([low - 2.0, high + 2.0]) / nyquist
        order, _ = cheb2ord(pass_band, stop_band, 3.0, 30.0)
        numerator, denominator = cheby2(order, 30.0, stop_band, btype="bandpass")
        np.testing.assert_allclose(stored["numerator"], numerator, rtol=0, atol=1e-12)
        np.testing.assert_allclose(stored["denominator"], denominator, rtol=0, atol=1e-12)
    assert "filter_frequency_response" in model.state_dict()


def test_fbmsnet_fft_filter_matches_released_causal_lfilter():
    rng = np.random.RandomState(7)
    signal = rng.randn(2, 3, 1001).astype(np.float32)
    model = FBMSNet8to32AT(3, 1001, 2, 250.0)
    with torch.no_grad():
        actual = model.causal_filter_bank(torch.from_numpy(signal[:, None])).numpy()
    expected = np.stack(
        [
            lfilter(item["numerator"], item["denominator"], signal, axis=-1)
            for item in model.filter_coefficients
        ],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("backbone_name", sorted(CORRECTED))
def test_corrected_networks_have_finite_nonzero_gradients(backbone_name):
    torch.manual_seed(3)
    backbone = registry.build(
        "models",
        backbone_name,
        n_chans=3,
        n_times=448 if backbone_name == "EEGWaveNetReleaseAT" else 1001,
        n_classes=2,
        sfreq=250.0,
    )
    model = PipelineModel(backbone, Linear(backbone.out_features, 2))
    x = torch.randn(2, 1, 3, backbone.n_times if hasattr(backbone, "n_times") else 1001)
    _, logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
