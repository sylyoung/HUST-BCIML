"""Cheap breadth coverage over every registered plug-in, plus the guards.

``test_smoke.py`` runs a handful of full leave-one-subject-out sweeps end to end,
which is slow, so it can only cover a sample of the catalogue. That left most of
the public leaderboard — the ten added backbones, the eight augmenters, the
fourteen ensemble combiners, the federated strategies — with no executable
coverage at all: they could break and the committed suite would still pass.

This module is the complementary half. Nothing here trains to convergence; each
test asserts that a plug-in constructs and produces correctly-shaped output at
the shapes the three benchmark datasets actually have. It runs in seconds and
catches the class of breakage (an import, a shape, a registry entry, a renamed
constructor argument) that a refactor introduces.

The second half of the file covers the composition guards and the results-writing
rules, which exist precisely to turn silent nonsense into an error — so they need
tests that assert the error actually fires.
"""
import numpy as np
import pytest
import torch

from hustbciml.core import registry
from hustbciml.core.batch import EEGBatch, EEGEpochs
from hustbciml.core.config import Config, resolve_config
from hustbciml.core.pipeline import build_pipeline
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject

# (n_chans, n_times, sfreq) of the three benchmark datasets, so a backbone is
# exercised at every shape it is actually published on rather than one of them.
DATASET_SHAPES = [(22, 1001, 250.0), (15, 2561, 512.0), (13, 2561, 512.0)]


@pytest.mark.parametrize("name", registry.available("models"))
@pytest.mark.parametrize("shape", DATASET_SHAPES, ids=["2014001", "2014002", "2015001"])
def test_backbone_builds_and_forwards(name, shape):
    n_chans, n_times, sfreq = shape
    torch.manual_seed(0)
    model = registry.resolve("models", name)(
        n_chans=n_chans, n_times=n_times, n_classes=2, sfreq=sfreq,
        F1=4, D=2, F2=8, dropout=0.25)
    assert model.out_features > 0, f"{name} did not set out_features"
    feats = model.forward_features(torch.zeros(2, 1, n_chans, n_times))
    assert feats.shape == (2, model.out_features), (name, feats.shape)


@pytest.mark.parametrize("name", registry.available("models"))
def test_backbone_shape_probe_leaves_no_trace(name):
    """Construction must not move BatchNorm statistics or the RNG.

    A dummy forward under a bare ``torch.no_grad()`` folds the probe tensor into
    every BatchNorm's running estimates and draws from the global RNG, so two
    backbones compared at the same seed would start from different random states
    purely because they probe differently — under a leaderboard whose premise is
    that rows differ in exactly one stage.
    """
    torch.manual_seed(0)
    rng_before = torch.random.get_rng_state()
    model = registry.resolve("models", name)(
        n_chans=22, n_times=1001, n_classes=2, sfreq=250.0, F1=4, D=2, F2=8, dropout=0.25)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            assert torch.allclose(m.running_mean, torch.zeros_like(m.running_mean)), name
            assert torch.allclose(m.running_var, torch.ones_like(m.running_var)), name
            assert int(m.num_batches_tracked) == 0, name
    assert model.training, f"{name} was left in eval mode after construction"
    del rng_before   # the RNG *is* consumed by parameter init; only the probe must not


@pytest.mark.parametrize("name", registry.available("ensembles"))
def test_combiner_returns_one_label_per_trial(name):
    """Every combiner fuses (K, N, C) scores into (N,) labels, and does better
    than the worst member on an easy synthetic vote table."""
    rng = np.random.RandomState(0)
    K, N, C = 5, 120, 2
    truth = rng.randint(0, C, N)
    scores = np.zeros((K, N, C))
    for k in range(K):
        acc = 0.55 + 0.08 * k
        pred = np.where(rng.rand(N) < acc, truth, 1 - truth)
        scores[k, np.arange(N), pred] = 1.0
        scores[k] += 0.01 * rng.rand(N, C)
    labels = registry.resolve("ensembles", name)().combine(scores)
    assert labels.shape == (N,), (name, labels.shape)
    assert set(np.unique(labels)) <= set(range(C)), (name, np.unique(labels))
    assert (labels == truth).mean() > 0.55, (name, (labels == truth).mean())


@pytest.mark.parametrize("name", registry.available("augmenters"))
def test_augmenter_preserves_the_batch_contract(name):
    """An augmenter returns a batch whose x/y/domain lengths agree and whose
    trials keep their (1, C, T) shape."""
    torch.manual_seed(0)
    C, T = 8, 128
    ch_names = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CPz"]
    aug = registry.resolve("augmenters", name)(
        ch_names=ch_names, n_classes=2, sfreq=128.0,
        classes=["left_hand", "right_hand"])
    x = torch.randn(16, 1, C, T)
    batch = EEGBatch(x, torch.tensor([0, 1] * 8), torch.tensor([0] * 8 + [1] * 8))
    out = aug(batch)
    assert out.x.shape[1:] == (1, C, T), (name, out.x.shape)
    assert out.x.shape[0] == out.y.shape[0] == out.domain.shape[0], name
    assert out.x.shape[0] >= x.shape[0], (name, "an augmenter may only add trials")
    assert torch.isfinite(out.x).all(), name


def test_channel_reflection_refuses_a_non_anatomical_montage():
    """Fail closed rather than invent a mirror.

    BNCI2014002 exposes ``EEG1 ... EEG15``. The odd/even hemisphere rule of the
    10-20 system says nothing about those labels, so applying it produces an
    arbitrary sensor permutation — which, combined with the two-class label swap,
    is fabricated training data that still yields a leaderboard-looking number.
    """
    cls = registry.resolve("augmenters", "ChannelReflection")
    with pytest.raises(ValueError, match="electrode positions"):
        cls(ch_names=[f"EEG{i}" for i in range(1, 16)], n_classes=2,
            classes=["left_hand", "right_hand"])


def test_channel_reflection_refuses_a_non_left_right_task():
    """A midline reflection of a feet trial is still a feet trial, so the label
    swap must not fire on right-hand-vs-feet data (BNCI2014002/2015001)."""
    cls = registry.resolve("augmenters", "ChannelReflection")
    with pytest.raises(ValueError, match="left/right"):
        cls(ch_names=["C3", "Cz", "C4"], n_classes=2, classes=["feet", "right_hand"])


def _toy_dims(cfg):
    cfg.n_chans, cfg.n_times, cfg.n_classes, cfg.sfreq = 8, 128, 2, 128.0
    cfg.ch_names = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CPz"]
    cfg.classes = ["left_hand", "right_hand"]
    return cfg


def test_pipeline_rejects_a_non_online_aligner_under_tta():
    """``--aligner RA --strategy Tent`` used to run online *EA* and report it as
    RA. The composition is refused instead of silently substituted."""
    cfg = _toy_dims(Config(aligner="RA", strategy="Tent", backbone="EEGNet", head="Linear"))
    with pytest.raises(ValueError, match="supports_online"):
        build_pipeline(cfg)


def test_unknown_hp_key_is_rejected():
    """An unrecognised ``--hp`` key is applied by nobody, so a typo would produce
    a default-valued run wearing a tuned run's label."""
    with pytest.raises(KeyError, match="unknown --hp key"):
        resolve_config(["--dataset", "Toy", "--hp", "asfa_bta=0.3"])


def test_calib_ratio_is_rejected_until_implemented():
    with pytest.raises(NotImplementedError, match="calib_ratio"):
        resolve_config(["--dataset", "Toy", "--calib_ratio", "0.2"])


def test_preset_override_gets_its_own_run_identity():
    """A partly-overridden preset must not be filed under the plain preset's
    name, or it overwrites the genuine result and ``metrics.json`` mislabels it."""
    plain, _ = resolve_config(["--algorithm", "EA-EEGNet", "--dataset", "Toy"])
    over, _ = resolve_config(["--algorithm", "EA-EEGNet", "--dataset", "Toy",
                              "--backbone", "ShallowFBCSPNetAT"])
    assert plain.setting() != over.setting()
    assert "ShallowFBCSPNetAT" in over.setting()


def test_run_identity_distinguishes_augmenters():
    """The augmenter is a whole leaderboard table's worth of variation; two
    hand-composed runs differing only in it must not share a results folder."""
    a, _ = resolve_config(["--dataset", "Toy", "--augmenter", "Identity"])
    b, _ = resolve_config(["--dataset", "Toy", "--augmenter", "CSDA"])
    assert a.setting() != b.setting()


def test_results_refuse_to_overwrite_a_different_config(tmp_path):
    """Re-running the same config overwrites in place (runs stay resumable);
    re-running a *different* config into the same folder is refused, because that
    replaces one measurement with another under one label."""
    cfg = Config(dataset="Toy", algorithm="EA-EEGNet", results_dir=str(tmp_path),
                 device="cpu", epochs=1, batch_size=16, lr=1e-3)
    exp = Exp_CrossSubject(cfg)
    per = [{"accuracy": 70.0, "primary": 70.0}]
    exp.save_results(per, {"primary": {"mean": 70.0, "std": 0.0}})
    exp.save_results(per, {"primary": {"mean": 70.0, "std": 0.0}})     # same config: fine

    other = Exp_CrossSubject(Config(dataset="Toy", algorithm="EA-EEGNet",
                                    results_dir=str(tmp_path), device="cpu",
                                    epochs=1, batch_size=16, lr=5e-4))
    with pytest.raises(FileExistsError, match="different measurement identity"):
        other.save_results(per, {"primary": {"mean": 71.0, "std": 0.0}})


def test_metrics_json_is_valid_json_with_a_nan_metric(tmp_path):
    """A NaN AUC (a fold in which one class never appears) is legitimate, but the
    default ``json.dump`` writes the bare token ``NaN``, which is not JSON and
    breaks every non-Python reader of the results tree."""
    import json

    cfg = Config(dataset="Toy", algorithm="EA-EEGNet", results_dir=str(tmp_path),
                 device="cpu")
    out = Exp_CrossSubject(cfg).save_results(
        [{"auc": float("nan"), "primary": 70.0}],
        {"auc": {"mean": float("nan"), "std": float("nan")},
         "primary": {"mean": 70.0, "std": 0.0}})
    with open(f"{out}/metrics.json") as fh:
        payload = json.load(fh)                # would raise on a bare NaN token
    assert payload["summary"]["auc"]["mean"] is None
    assert payload["config"]["seed"] == cfg.seed


def test_metrics_json_records_the_full_config(tmp_path):
    """A leaderboard cell has to be auditable back to the settings that produced
    it from the artifact alone; the folder name cannot carry them all."""
    import json

    cfg = Config(dataset="Toy", algorithm="EA-EEGNet", results_dir=str(tmp_path),
                 device="cpu", lr=3e-4, epochs=7, hp={"asfa_beta": 0.3})
    out = Exp_CrossSubject(cfg).save_results(
        [{"primary": 70.0}], {"primary": {"mean": 70.0, "std": 0.0}})
    with open(f"{out}/metrics.json") as fh:
        payload = json.load(fh)
    assert payload["config"]["lr"] == 3e-4
    assert payload["config"]["epochs"] == 7
    assert payload["config"]["hp"] == {"asfa_beta": 0.3}


@pytest.mark.parametrize("strategy", ["ERM", "MDMAML"])
def test_training_estimates_the_batchnorm_running_statistics(strategy):
    """A strategy that reverts weights between inner steps must not also revert the
    BatchNorm buffers.

    In train mode BatchNorm normalises with the *batch* statistics, so the running
    buffers never enter the forward pass and cannot affect any gradient — but
    ``predict`` runs in eval mode and scores against exactly those buffers. A
    strategy that restores them after every inner step therefore leaves them at
    their initialisation (mean 0, var 1) forever, and the model gets evaluated
    under a normalisation it was never trained with. MDMAML shipped that way and
    lost ~16 accuracy points on BNCI2014001 before this test existed.
    """
    from hustbciml.core.context import RunContext

    cfg = _toy_dims(Config(dataset="Toy", device="cpu", strategy=strategy, epochs=2))
    pipe = build_pipeline(cfg)
    rng = np.random.RandomState(0)
    # >=2 domains: MDMAML pairs source domains and falls back to plain supervised
    # training with fewer, which would not exercise the inner-loop revert at all.
    epochs = EEGEpochs(X=rng.randn(64, 8, 128).astype(np.float32),
                       y=rng.randint(0, 2, 64).astype(np.int64),
                       domain=np.repeat([0, 1, 2, 3], 16).astype(np.int64),
                       sfreq=128.0, n_classes=2)
    ctx = RunContext(cfg=cfg, device=torch.device("cpu"), augmenter=pipe.augmenter,
                     aligner=pipe.aligner, log=lambda m: None, target_unlabeled=None)

    before = {n: b.detach().clone() for n, b in pipe.model.named_buffers()}
    assert before, "backbone has no BatchNorm buffers — this test would prove nothing"
    pipe.strategy.fit(pipe.model, epochs, ctx)
    after = dict(pipe.model.named_buffers())

    moved = [n for n, b in before.items()
             if "running_" in n and not torch.allclose(b, after[n])]
    assert moved, (
        f"{strategy}: every BatchNorm running statistic is still at its "
        f"initialisation after training, so eval-mode inference will normalise "
        f"with mean 0 / var 1")


def test_untrained_model_is_not_scored():
    """``--epochs 0`` (or a batch size larger than the training split) leaves the
    network at its random initialisation. Publishing a number for that is worse
    than failing."""
    from hustbciml.algorithms.strategies._common import supervised_train
    from hustbciml.core.context import RunContext

    cfg = _toy_dims(Config(dataset="Toy", device="cpu", epochs=0))
    pipe = build_pipeline(cfg)
    epochs = EEGEpochs(X=np.zeros((40, 8, 128), dtype=np.float32),
                       y=np.zeros(40, dtype=np.int64),
                       domain=np.zeros(40, dtype=np.int64), sfreq=128.0, n_classes=2)
    ctx = RunContext(cfg=cfg, device=torch.device("cpu"), augmenter=pipe.augmenter,
                     aligner=pipe.aligner, log=lambda m: None, target_unlabeled=None)
    with pytest.raises(RuntimeError, match="0 optimizer steps"):
        supervised_train(pipe.model, epochs, ctx)
