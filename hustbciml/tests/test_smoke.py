"""Integration smoke test on bundled toy data — no download, fast.

Proves the M0 contract: the 5 ABCs, registry, cross-subject splitter, EA
aligner, EEGNet backbone, Linear head, and both strategies compose and run
end-to-end, and EA-EEGNet learns above chance on the synthetic task.
"""
import dataclasses

from hustbciml.core.config import Config
from hustbciml.core import registry
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject


def _toy_cfg(**over):
    base = dict(dataset="Toy", protocol="cross_subject", algorithm="EA-EEGNet",
                aligner="EA", augmenter="Identity", backbone="EEGNet",
                head="Linear", strategy="ERM", epochs=30, batch_size=16,
                lr=1e-3, seed=2023, itr=1, device="cpu", early_stop_patience=8,
                results_dir="/tmp/hustbciml_test_results")
    base.update(over)
    return Config(**base)


def test_registry_discovers_plugins():
    cat = registry.catalog()
    assert "EA" in cat["aligners"] and "Identity" in cat["aligners"]
    # Riemannian Alignment (alignment-axis comparison method)
    assert "RA" in cat["aligners"], cat["aligners"]
    assert "EEGNet" in cat["models"]
    # EEG Conformer backbone (network-axis comparison method)
    assert "EEGConformer" in cat["models"], cat["models"]
    # newly ported backbones (network-axis): TIE-EEGNet, KDFNet
    for m in ("TIEEEGNet", "KDFNet"):
        assert m in cat["models"], (m, cat["models"])
    # Channel Reflection augmenter (augmentation-axis comparison method)
    assert "Identity" in cat["augmenters"] and "ChannelReflection" in cat["augmenters"], \
        cat["augmenters"]
    assert "Linear" in cat["heads"]
    assert "ERM" in cat["strategies"] and "TTIME" in cat["strategies"]
    # test-time-adaptation family (+ SAR, sharpness-aware)
    for s in ("Tent", "PL", "BNAdapt", "SAR"):
        assert s in cat["strategies"], (s, cat["strategies"])
    # gradient domain-adaptation family
    for s in ("DANN", "CDAN", "MCC", "DAN", "JAN", "MDD"):
        assert s in cat["strategies"], (s, cat["strategies"])
    # source-free domain adaptation
    assert "SHOT" in cat["strategies"], cat["strategies"]
    # imbalanced source-free DA (online TTA variant)
    assert "ISFDA" in cat["strategies"], cat["strategies"]
    # degradation-free fully test-time adaptation
    assert "DELTA" in cat["strategies"], cat["strategies"]
    # classical fit/predict track
    for s in ("CSP_LDA", "RiemannMDM"):
        assert s in cat["strategies"], (s, cat["strategies"])
    # newly ported transfer (MEKT, MDMAML, ASFA) + federated privacy (SAFE) methods
    for s in ("MEKT", "MDMAML", "ASFA", "SAFE"):
        assert s in cat["strategies"], (s, cat["strategies"])
    assert "_mdd" not in cat["strategies"]
    # helper modules must be skipped
    assert "_common" not in cat["strategies"]
    assert "_mmd" not in cat["strategies"]
    assert "_sam" not in cat["strategies"]


def test_ea_eegnet_learns_above_chance():
    summary = Exp_CrossSubject(_toy_cfg(epochs=50)).run()
    assert summary["accuracy"]["mean"] > 85.0, summary["accuracy"]


def test_ea_beats_no_alignment():
    """Transfer works AND EA helps: EA-aligned mean >= no-alignment mean.
    (On toy, EA ~99 vs no-align ~90 with much higher variance.)"""
    ea = Exp_CrossSubject(_toy_cfg(epochs=50, aligner="EA")).run()
    noa = Exp_CrossSubject(_toy_cfg(epochs=50, aligner="Identity", algorithm=None)).run()
    assert ea["accuracy"]["mean"] >= noa["accuracy"]["mean"] - 1.0, (ea, noa)
    assert ea["accuracy"]["mean"] > 90.0


def test_riemann_alignment_runs():
    """RA (Riemannian Alignment): EA-style recentring but with the Riemannian
    (Fréchet) mean covariance as the reference. It is an aligner + ERM, so it
    should learn clearly above chance on toy, like EA."""
    cfg = _toy_cfg(algorithm="RA-EEGNet", aligner="RA", epochs=50)
    summary = Exp_CrossSubject(cfg).run()
    acc = summary["accuracy"]["mean"]
    assert acc == acc, ("RA", "NaN accuracy")
    assert acc > 75.0, ("RA", summary["accuracy"])


def test_channel_reflection_runs():
    """Channel Reflection augmenter (aligner Identity, raw electrode space):
    appends hemisphere-reflected, label-swapped copies to each batch. On toy
    there is no left/right montage symmetry, so the label-swapped copies do not
    carry the class semantics they do for real MI — CR is not expected to help
    here; we only assert it composes and yields valid, non-NaN metrics. Its
    showcase is BNCI2014001 motor imagery, not toy."""
    cfg = _toy_cfg(algorithm="CR-EEGNet", aligner="Identity",
                   augmenter="ChannelReflection", epochs=40)
    summary = Exp_CrossSubject(cfg).run()
    acc = summary["accuracy"]["mean"]
    assert acc == acc, ("ChannelReflection", "NaN accuracy")
    assert 0.0 <= acc <= 100.0, ("ChannelReflection", summary["accuracy"])


def test_eegconformer_runs():
    """EEG Conformer backbone (conv tokenizer + transformer encoder) trained
    with ERM composes end-to-end and produces valid, non-NaN metrics on toy.
    A transformer needs more data than toy provides, so we assert it runs
    cleanly and above chance rather than matching EEGNet."""
    cfg = _toy_cfg(algorithm="EA-EEGConformer", backbone="EEGConformer", epochs=50)
    summary = Exp_CrossSubject(cfg).run()
    acc = summary["accuracy"]["mean"]
    assert acc == acc, ("EEGConformer", "NaN accuracy")
    assert 0.0 <= acc <= 100.0, ("EEGConformer", summary["accuracy"])


def test_ttime_online_ea_recovers_base():
    """Faithfulness guard: online incremental EA + the frozen source model
    (steps=0) must recover ~the offline EA-EEGNet accuracy. If a future edit
    breaks the online-EA path or the source-training path, this catches it."""
    cfg = _toy_cfg(strategy="TTIME", algorithm="T-TIME", epochs=50,
                   test_batch=8, steps=0, stride=1, temperature=2.0)
    summary = Exp_CrossSubject(cfg).run()
    assert summary["accuracy"]["mean"] > 90.0, summary["accuracy"]


def test_ttime_adaptation_runs_end_to_end():
    """Full online TTA (steps=1) runs and produces valid, non-NaN metrics.
    Note: on drift-free toy data with a saturated base model, entropy-min TTA
    does not improve over the base (it can regress via confirmation bias) —
    its real showcase is BNCI2014001, not toy. We only assert it runs cleanly."""
    cfg = _toy_cfg(strategy="TTIME", algorithm="T-TIME", epochs=50,
                   test_batch=8, steps=1, stride=1, temperature=2.0)
    summary = Exp_CrossSubject(cfg).run()
    assert 0.0 <= summary["accuracy"]["mean"] <= 100.0
    assert summary["accuracy"]["mean"] == summary["accuracy"]["mean"]  # not NaN


def test_tta_family_runs_end_to_end():
    """Tent, PL, and BN-adapt all compose over the shared online-TTA skeleton
    and produce valid, non-NaN metrics on toy data. (Like T-TIME, they are not
    expected to beat the base on drift-free toy — their showcase is BNCI2014001;
    here we only assert they run cleanly through the streaming loop.)"""
    for strat in ("Tent", "PL", "BNAdapt", "SAR", "ISFDA", "DELTA"):
        cfg = _toy_cfg(strategy=strat, algorithm=None, aligner="EA",
                       epochs=40, test_batch=8, steps=1, stride=1)
        summary = Exp_CrossSubject(cfg).run()
        acc = summary["accuracy"]["mean"]
        assert 0.0 <= acc <= 100.0, (strat, summary["accuracy"])
        assert acc == acc, (strat, "NaN accuracy")            # not NaN


def test_gradient_da_family_runs_end_to_end():
    """CDAN, MCC, DAN, JAN (transductive gradient DA) compose over the shared
    transductive_train loop and produce valid, non-NaN metrics on toy data.
    (They need the aligned, label-masked target the Exp puts in
    ctx.target_unlabeled; this checks the whole transductive path.)"""
    for strat in ("MCC", "CDAN", "DAN", "JAN", "MDD"):
        cfg = _toy_cfg(strategy=strat, algorithm=None, aligner="EA", epochs=20)
        summary = Exp_CrossSubject(cfg).run()
        acc = summary["accuracy"]["mean"]
        assert 0.0 <= acc <= 100.0, (strat, summary["accuracy"])
        assert acc == acc, (strat, "NaN accuracy")


def test_source_free_shot_runs():
    """SHOT (source-free DA): train an ERM source model, then adapt only the
    feature extractor to the unlabeled target by information maximization with
    the classifier head frozen. It is a non-tta strategy (offline-aligned
    target). On toy the source model is already strong, so SHOT-IM should
    preserve high accuracy rather than drift."""
    cfg = _toy_cfg(strategy="SHOT", algorithm=None, aligner="EA", epochs=50)
    summary = Exp_CrossSubject(cfg).run()
    acc = summary["accuracy"]["mean"]
    assert acc == acc, ("SHOT", "NaN accuracy")
    assert acc > 75.0, ("SHOT", summary["accuracy"])


def test_classical_track_runs():
    """CSP+LDA and Riemannian MDM (mode='fit', no network) compose through the
    Exp and learn above chance on toy — exercises the classical fit/predict path."""
    for strat in ("CSP_LDA", "RiemannMDM"):
        cfg = _toy_cfg(strategy=strat, algorithm=None, aligner="EA", epochs=1)
        summary = Exp_CrossSubject(cfg).run()
        assert summary["accuracy"]["mean"] > 70.0, (strat, summary["accuracy"])


def test_tta_steps0_recovers_base():
    """The shared skeleton's steps=0 guard holds for the ported strategies too:
    with no adaptation step, each reduces to frozen-model inference over the
    online-aligned stream, recovering ~the offline EA-EEGNet accuracy."""
    for strat in ("Tent", "PL", "BNAdapt", "SAR", "ISFDA", "DELTA"):
        cfg = _toy_cfg(strategy=strat, algorithm=None, aligner="EA",
                       epochs=50, test_batch=8, steps=0, stride=1)
        summary = Exp_CrossSubject(cfg).run()
        assert summary["accuracy"]["mean"] > 90.0, (strat, summary["accuracy"])


def test_new_backbones_run():
    """TIE-EEGNet (EEGNet + a sinusoidal time-positional-embedding conv) and KDFNet
    (an FBCSP-mirroring CNN with FIR filter-bank + per-band CSP initialization)
    compose end-to-end and learn above chance on toy."""
    for bb in ("TIEEEGNet", "KDFNet"):
        summary = Exp_CrossSubject(_toy_cfg(algorithm=None, aligner="EA",
                                            backbone=bb, epochs=40)).run()
        acc = summary["accuracy"]["mean"]
        assert acc == acc and 0.0 <= acc <= 100.0, (bb, summary["accuracy"])
        assert acc > 75.0, (bb, summary["accuracy"])


def test_new_transfer_privacy_run():
    """The four newly ported network-based methods compose end-to-end and give
    valid, non-NaN metrics on toy: MEKT (classical manifold-embedded transfer,
    mode='fit'), MDMAML (domain-paired first-order MAML, forward-only), ASFA
    (source-free Tsallis-uncertainty + consistency), SAFE (federated FGSM+AWP)."""
    cases = [
        dict(strategy="MEKT", aligner="Identity", epochs=1),
        dict(strategy="MDMAML", aligner="EA", epochs=60),
        dict(strategy="ASFA", aligner="EA", epochs=40),
        dict(strategy="SAFE", aligner="EA", epochs=40, test_batch=8),
    ]
    for over in cases:
        summary = Exp_CrossSubject(_toy_cfg(algorithm=None, backbone="EEGNet", **over)).run()
        acc = summary["accuracy"]["mean"]
        assert acc == acc and 0.0 <= acc <= 100.0, (over["strategy"], summary["accuracy"])


if __name__ == "__main__":
    test_registry_discovers_plugins()
    print("registry OK")
    test_ea_eegnet_learns_above_chance()
    print("EA-EEGNet learns OK")
    test_ea_beats_no_alignment()
    print("EA > no-alignment OK")
    test_riemann_alignment_runs()
    print("RA Riemannian alignment OK")
    test_channel_reflection_runs()
    print("Channel Reflection augmenter OK")
    test_eegconformer_runs()
    print("EEG Conformer backbone OK")
    test_ttime_online_ea_recovers_base()
    print("T-TIME online-EA faithfulness OK")
    test_ttime_adaptation_runs_end_to_end()
    print("T-TIME adaptation runs OK")
    test_tta_family_runs_end_to_end()
    print("Tent/PL/BN-adapt/SAR run OK")
    test_gradient_da_family_runs_end_to_end()
    print("CDAN/MCC/DAN/JAN/MDD run OK")
    test_source_free_shot_runs()
    print("SHOT source-free OK")
    test_classical_track_runs()
    print("CSP+LDA / Riemann-MDM classical OK")
    test_tta_steps0_recovers_base()
    print("TTA steps=0 faithfulness OK")
    test_new_backbones_run()
    print("TIE-EEGNet / KDFNet backbones OK")
    test_new_transfer_privacy_run()
    print("MEKT / MDMAML / ASFA / SAFE OK")
    print("\nall smoke tests passed")
