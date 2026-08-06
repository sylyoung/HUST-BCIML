"""Single reportable inventory for the Network benchmark and its publication tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class NetworkMethod:
    display_name: str
    public_key: str
    backbone: str
    reference_basis: str
    scope: str = "two_class_mi_architecture_transfer"
    aligner: str = "EA"
    augmenter: str = "Identity"
    head: str = "Linear"
    strategy: str = "ERM"
    input_view: str = "moabb_8_32_hz"
    objective: str = "cross_entropy"
    n_classes: int = 2


NETWORK_METHODS = (
    NetworkMethod(
        "EEGNet",
        "EA-EEGNet-Nested",
        "EEGNet",
        "released architecture transfer under the Network nested-selection contract",
    ),
    NetworkMethod(
        "ShallowFBCSPNet-AT",
        "EA-ShallowFBCSPNet-AT",
        "ShallowFBCSPNetAT",
        "Braindecode f7562e9 feature architecture",
    ),
    NetworkMethod(
        "Deep4Net-AT",
        "EA-Deep4Net-AT",
        "Deep4NetAT",
        "Braindecode f7562e9 feature architecture",
    ),
    NetworkMethod(
        "EEGConformer", "EA-EEGConformer", "EEGConformer", "released architecture transfer"
    ),
    NetworkMethod(
        "DBConformer", "EA-DBConformer", "DBConformer", "released architecture transfer"
    ),
    NetworkMethod("CSP-Net", "CSP-Net", "CSPNet", "lab architecture transfer"),
    NetworkMethod("TIE-EEGNet", "EA-TIEEEGNet", "TIEEEGNet", "lab architecture transfer"),
    NetworkMethod("KDFNet", "EA-KDFNet", "KDFNet", "lab architecture transfer"),
    NetworkMethod(
        "ADFCNN-Transpose-AT",
        "EA-ADFCNN-Transpose-AT",
        "ADFCNNTransposeAT",
        "released feature architecture with corrected attention transpose",
    ),
    NetworkMethod("CTNet", "EA-CTNet", "CTNet", "released architecture transfer"),
    NetworkMethod("MSCFormer", "EA-MSCFormer", "MSCFormer", "released architecture transfer"),
    NetworkMethod("MSVTNet", "EA-MSVTNet", "MSVTNet", "released architecture transfer"),
    NetworkMethod("TMSA-Net", "EA-TMSANet", "TMSANet", "released architecture transfer"),
    NetworkMethod(
        "EEGWaveNet-Release-AT",
        "EA-EEGWaveNet-Release-AT",
        "EEGWaveNetReleaseAT",
        "released code 3b19098 feature architecture",
    ),
    NetworkMethod("SlimSeiz", "EA-SlimSeiz", "SlimSeiz", "released architecture transfer"),
    NetworkMethod(
        "FBMSNet-8-32-AT",
        "EA-FBMSNet-8-32-AT",
        "FBMSNet8to32AT",
        "released code 1c6b659 adapted to six causal 8-32 Hz views",
    ),
    NetworkMethod("EEGNeX", "EA-EEGNeX", "EEGNeX", "released architecture transfer"),
    NetworkMethod(
        "EEG-Deformer", "EA-EEGDeformer", "EEGDeformer", "released architecture transfer"
    ),
)

NETWORK_METHOD_BY_NAME = {method.display_name: method for method in NETWORK_METHODS}
NETWORK_METHOD_BY_KEY = {method.public_key: method for method in NETWORK_METHODS}

if len(NETWORK_METHOD_BY_NAME) != len(NETWORK_METHODS):
    raise RuntimeError("Network method display names must be unique")
if len(NETWORK_METHOD_BY_KEY) != len(NETWORK_METHODS):
    raise RuntimeError("Network method public keys must be unique")


def select_network_methods(names: Iterable[str] | None = None) -> tuple[NetworkMethod, ...]:
    """Resolve an ordered, duplicate-free display-name request."""
    if names is None:
        return NETWORK_METHODS
    requested = list(names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("network method request must be non-empty and contain no duplicates")
    unknown = sorted(set(requested) - set(NETWORK_METHOD_BY_NAME))
    if unknown:
        raise KeyError(
            f"unknown Network methods {unknown}; available: {sorted(NETWORK_METHOD_BY_NAME)}"
        )
    return tuple(NETWORK_METHOD_BY_NAME[name] for name in requested)


def network_method_manifest() -> list[dict]:
    """Return the complete JSON-safe reportable method manifest."""
    return [asdict(method) for method in NETWORK_METHODS]
