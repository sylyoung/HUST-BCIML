# ===========================================================================
# BFT.py  —  HUST-BCIML EEG-decoding benchmark
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.  Part of the unified benchmark; see repo README.
#
# Reference (IEEE BibTeX):
#   @Article{Li2026,
#     author  = {Li, Siyang and Ouyang, Jiayi and Cui, Zhenyao and Wang, Ziwei and Jia, Tianwang and Wan, Feng and Wu, Dongrui},
#     journal = {IEEE Journal of Biomedical and Health Informatics},
#     title   = {Backpropagation-Free Test-Time Adaptation for Lightweight {EEG}-Based Brain-Computer Interfaces},
#     year    = {2026},
#     note    = {arXiv:2601.07556},
#   }
# ===========================================================================
"""BFT — Backpropagation-Free Test-time adaptation (Li et al., 2026), BFT-A variant.

Faithful port of the authors' released **BFT-A** pipeline (``BFT-classify/``:
``train_pre_model.py``, ``train_loss_model.py``, ``augment.py``, ``test.py``,
``models/{EEGNet,losspredictor}.py``). The method has three trained pieces and a
backprop-free test phase — all reproduced here:

1. **Source model, trained WITH augmentation, on (t-1)-second trials.**
   BFT's decoder ``g,h`` (EEGNet) is trained on the Euclidean-aligned source with
   the paper's on-the-fly augmentation (Sec. V-C): each trial is randomly replaced
   by ONE of ~14 knowledge-guided transforms per epoch — the K=12 label-preserving
   views below plus the two label-aware ones, Channel Reflection (left/right-hand
   only) and DWT/CSDA (same-class detail graft). Crucially,
   every trial is truncated to ``eeg_length = (round(T/sfreq) - 1) * sfreq``
   samples — the model runs on (t-1)-second inputs so that at test time a full
   (t-1)-second window can *slide* within the t-second trial. This truncation is
   intrinsic to BFT, so BFT trains its own model rather than sharing the
   benchmark's full-length EA-EEGNet source (disclosed below).

2. **Reliability / ranking module, trained on the source.** An
   ``EEGNetLossPredictor`` (3-layer MLP on the flattened backbone features, the
   authors' ``models/losspredictor.py``) is trained so that, over the K=12
   transformations of a trial, ``softmax(-predicted_loss)`` rank-matches
   ``softmax(-real_cross_entropy)`` — reliable (low-loss) views earn more weight.

3. **Backprop-free test phase (the K=12 real transformations).** Each streamed
   target trial is online-Euclidean-aligned, then K=12 label-preserving views are
   built and predicted; the temperature-sharpened (tau=0.25) softmaxes are
   averaged with per-view reliability weights (accumulated as a running mean over
   the stream). No gradient is taken; the only adaptation is refreshing the
   BatchNorm running statistics on a sliding window of the most recent trials.

The K=12 views (authors' ``generate_augmented_inputs``), each returned at
``eeg_length`` samples: identity; additive uniform noise (per-trial std / 2);
amplitude scaling x0.9, x1.1, x1.2; Hilbert analytic frequency shift +/-0.2 Hz;
and **five real sliding windows** cropped from the full t-second trial at onsets
stride*{1..5} (stride = 0.2 s) — the genuine (t-1)-second segments the method
relies on for view diversity, not zero-padded shifts. Aggregation temperature
tau = 0.25.

Faithful-adaptation disclosures (research integrity):
  (1) Capacity: this port uses the benchmark's shared EEGNet capacity
      (F1=4/D=2/F2=8) so BFT is comparable to every other backbone/TTA row; the
      paper uses F1=8/D=2/F2=16. The mechanism (truncated multi-view + reliability
      aggregation + BN refresh) is otherwise the authors'.
  (2) The reliability predictor in the release is trained with a SoDeep
      differentiable-sorting Spearman loss whose *pretrained sorter weights are
      not shipped*. It is replaced here by a soft-rank Spearman surrogate with the
      same rank-correlation objective (``_soft_spearman_loss``); the predictor
      architecture, inputs, and training target are otherwise the authors'.
  (3) Training augmentation is the paper's on-the-fly random mix of 14 transforms
      (Identity, Scale x3, Noise, Freq x2, Slide x5, plus the label-aware Channel
      Reflection and DWT/CSDA), one drawn per trial per epoch. Channel Reflection is
      dropped on datasets that are not left/right-hand (BNCI2014002/2015001 are
      right-hand-vs-feet), where its electrode mirror + label swap would poison
      training — matching the benchmark's rule that CR is a two-class left/right
      transform. The BFT-D dropout variant is not ported (BFT-A only).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hustbciml.algorithms.aligners.EA import EA
from hustbciml.core.batch import EEGEpochs
from hustbciml.core.context import RunContext
from hustbciml.core.stages import Strategy
from hustbciml.utils.metrics import accuracy
from hustbciml.utils.montage import reflection_permutation
from hustbciml.utils.tools import EarlyStopping
from ._common import forward_logits, split_train_val


# --------------------------------------------------------------------------
# Helpers: (t-1)-second length and the 12 label-preserving views.
# --------------------------------------------------------------------------
def _eeg_length(n_times: int, sfreq: float) -> int:
    """Authors' truncation: (round(T/sfreq) - 1) * sfreq samples = (t-1) seconds.

    Falls back to the full length for trials shorter than ~2 s (where a (t-1)-s
    window would vanish), so the strategy stays runnable on any dataset; the
    sliding-window views then degrade to the identity (disclosed at runtime)."""
    sec = int(round(n_times / float(sfreq)))
    eeg_length = int((sec - 1) * sfreq)
    if eeg_length <= 0 or eeg_length >= n_times:
        return int(n_times)
    return eeg_length


def _freqshift(X: np.ndarray, sfreq: float, f_shift: float) -> np.ndarray:
    """Hilbert-analytic frequency shift by ``f_shift`` Hz over (N, C, T)."""
    from scipy.signal import hilbert
    N, C, T = X.shape
    dt = 1.0 / float(sfreq)
    padlen = 2 ** int(np.ceil(np.log2(T)))
    shift = np.exp(2j * np.pi * f_shift * dt * np.arange(padlen))
    padded = np.concatenate([X, np.zeros((N, C, padlen - T))], axis=2)
    analytic = hilbert(padded, axis=2)
    return (analytic * shift[None, None, :]).real[:, :, :T].astype(X.dtype)


def _bft_views(X_full: np.ndarray, sfreq: float, eeg_length: int, seed: int) -> List[np.ndarray]:
    """K=12 views of the FULL trial ``X_full`` (N, C, T), each returned at
    ``eeg_length`` samples, matching the authors' ``augment.py``:

    identity / noise / scale{.9,1.1,1.2} / freq{+.2,-.2} are computed on the full
    trial then truncated to the first ``eeg_length`` samples; the five sliding
    windows are real crops ``[start : start + eeg_length]`` at onset stride*no.
    """
    N, C, T = X_full.shape
    L = int(eeg_length)
    rng = np.random.RandomState(seed)
    head = lambda A: A[:, :, :L]                              # first (t-1) s

    views = [head(X_full)]                                    # 1. identity
    std = X_full.std(axis=(1, 2), keepdims=True)              # per-trial scalar amplitude
    views.append(head(X_full + (rng.rand(*X_full.shape) - 0.5) * std / 2.0))  # 2. noise
    for a in (0.9, 1.1, 1.2):                                 # 3-5. amplitude scaling
        views.append(head(X_full * a))
    views.append(head(_freqshift(X_full, sfreq, 0.2)))        # 6. high-frequency shift
    views.append(head(_freqshift(X_full, sfreq, -0.2)))       # 7. low-frequency shift
    stride = max(1, int(sfreq * 0.2))                         # 8-12. five sliding windows
    for no in (1, 2, 3, 4, 5):
        start = min(stride * no, max(0, T - L))               # clamp so the window fits
        views.append(X_full[:, :, start:start + L])
    return views                                             # len 12, each (N, C, L)


# --------------------------------------------------------------------------
# Training augmentation (paper Sec. V-C): on-the-fly, each source trial is
# randomly replaced by ONE of ~14 knowledge-guided transforms per epoch. The
# label-preserving ones match the K=12 test-time views; the two label-AWARE ones
# are Channel Reflection (left/right-hand only — it mirrors electrodes and swaps
# the L<->R label) and DWT/CSDA (same-class high-frequency detail graft). This
# operates on FULL-length trials and returns (t-1)-second samples, so the sliding
# windows are real crops.
# --------------------------------------------------------------------------
def _is_left_right(classes) -> bool:
    """True iff the two class names are a left/right-hand pair — the only case in
    which Channel Reflection's electrode mirror maps one class onto the other and
    the label swap is valid. ``['left_hand','right_hand']`` -> True;
    ``['feet','right_hand']`` (BNCI2014002/2015001) -> False, so CR is dropped."""
    names = [str(c).lower() for c in (classes or [])]
    if len(names) != 2:
        return False
    return any("left" in n for n in names) and any("right" in n for n in names)


def _dwt_swap(Xb: np.ndarray, yb: np.ndarray, idx: np.ndarray, L: int,
              rng: np.random.RandomState, wavelet: str = "db4",
              mode: str = "smooth") -> np.ndarray:
    """DWT/CSDA (Ziwei Wang et al., 2025) for the batch rows ``idx``: keep each
    trial's low-frequency approximation, graft a random SAME-CLASS batch partner's
    high-frequency detail. Falls back to the identity (t-1)s crop when a class is a
    singleton in the batch or PyWavelets is unavailable."""
    head = Xb[:, :, :L].astype(np.float64)                   # the (t-1)s window
    out = head[idx].copy()
    try:
        import pywt
    except ImportError:                                      # pragma: no cover
        return out.astype(np.float32)
    for j, i in enumerate(idx):
        same = np.where(yb == yb[i])[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        p = same[rng.randint(len(same))]
        cA, _ = pywt.dwt(head[i], wavelet, axis=-1)
        _, pD = pywt.dwt(head[p], wavelet, axis=-1)
        rec = pywt.idwt(cA, pD, wavelet, mode, axis=-1)
        m = min(rec.shape[-1], L)
        out[j, :, :m] = rec[..., :m]
    return out.astype(np.float32)


def _bft_train_augment(Xb: np.ndarray, yb: np.ndarray, sfreq: float, L: int,
                       ch_perm, use_cr: bool, rng: np.random.RandomState):
    """Assign each trial in the batch one random transform (equal probability) and
    return the augmented (t-1)-second batch ``(B, C, L)`` plus its labels — only the
    Channel-Reflection-assigned rows have their left/right label flipped."""
    B, C, T = Xb.shape
    out = np.empty((B, C, L), dtype=np.float32)
    yout = yb.astype(np.int64).copy()

    menu = ["id", "sc0.9", "sc1.1", "sc1.2", "noise", "fq+", "fq-",
            "sl1", "sl2", "sl3", "sl4", "sl5", "dwt"]         # 13 label-preserving/-agnostic
    if use_cr:
        menu = menu + ["cr"]                                 # + Channel Reflection = 14 (paper)
    tid = rng.randint(0, len(menu), size=B)
    stride = max(1, int(sfreq * 0.2))
    std = Xb.std(axis=(1, 2), keepdims=True)

    for k, name in enumerate(menu):
        idx = np.where(tid == k)[0]
        if len(idx) == 0:
            continue
        if name == "id":
            out[idx] = Xb[idx, :, :L]
        elif name[:2] == "sc":
            out[idx] = (Xb[idx] * float(name[2:]))[:, :, :L]
        elif name == "noise":
            nz = (rng.rand(len(idx), C, T) - 0.5) * std[idx] / 2.0
            out[idx] = (Xb[idx] + nz)[:, :, :L]
        elif name == "fq+":
            out[idx] = _freqshift(Xb[idx], sfreq, 0.2)[:, :, :L]
        elif name == "fq-":
            out[idx] = _freqshift(Xb[idx], sfreq, -0.2)[:, :, :L]
        elif name[:2] == "sl":
            start = min(stride * int(name[2:]), max(0, T - L))
            out[idx] = Xb[idx, :, start:start + L]
        elif name == "dwt":
            out[idx] = _dwt_swap(Xb, yb, idx, L, rng)
        elif name == "cr":
            out[idx] = Xb[idx][:, ch_perm, :L]
            yout[idx] = 1 - yout[idx]                         # left <-> right (2-class)
    return out, yout


def _bft_train(inner: nn.Module, source: EEGEpochs, ctx: RunContext,
               sfreq: float, eeg_length: int) -> nn.Module:
    """Train BFT's (t-1)-second decoder with the paper's on-the-fly random-mix
    augmentation, early-stopping on a held-out source split scored on the identity
    (t-1)s crop. Channel Reflection enters the transform menu only for left/right-
    hand data (``source.classes``); on right-hand-vs-feet data it is dropped, so the
    label swap can never poison a non-left/right task."""
    cfg, device = ctx.cfg, ctx.device
    L = int(eeg_length)
    use_cr = _is_left_right(getattr(source, "classes", None)) and int(cfg.n_classes) == 2
    ch_perm = None
    if use_cr:
        perm = reflection_permutation(list(source.ch_names))
        if len(perm) == 0:                                   # no montage -> no valid reflection
            use_cr = False
        else:
            ch_perm = perm
    ctx.log(f"  [BFT] training aug: {'14' if use_cr else '13'} transforms, "
            f"Channel Reflection {'ON (left/right-hand)' if use_cr else 'OFF (not left/right-hand)'}")

    tr_idx, va_idx = split_train_val(len(source), cfg.val_ratio, cfg.seed)
    has_val = len(va_idx) > 0
    Xtr = source.X[tr_idx].astype(np.float64)
    ytr = source.y[tr_idx].astype(np.int64)
    val_trunc = source.select(va_idx).with_X(source.X[va_idx][:, :, :L]) if has_val else None

    inner.to(device)
    opt = torch.optim.Adam([p for p in inner.parameters() if p.requires_grad],
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")
    C = source.n_channels
    bs = max(2, cfg.batch_size)
    Ntr = len(Xtr)

    for epoch in range(cfg.epochs):
        inner.train()
        rng = np.random.RandomState(cfg.seed + epoch)
        perm = rng.permutation(Ntr)
        for s in range(0, Ntr - bs + 1, bs):
            idx = perm[s:s + bs]
            Xa, ya = _bft_train_augment(Xtr[idx], ytr[idx], sfreq, L, ch_perm, use_cr, rng)
            xt = torch.from_numpy(Xa.reshape(len(idx), 1, C, L)).float().to(device)
            yt = torch.from_numpy(ya).long().to(device)
            _, logits = inner(xt)
            loss = ce(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if has_val:
            logits = forward_logits(inner, val_trunc, device)
            acc = accuracy(val_trunc.y, logits.argmax(1))
            stopper.step(acc, inner)
            if stopper.should_stop:
                ctx.log(f"  [BFT] early stop epoch {epoch + 1} (best val_acc={stopper.best:.2f})")
                break
    if has_val:
        stopper.restore(inner)
        setattr(inner, "_val_score", float(stopper.best))
    return inner


# --------------------------------------------------------------------------
# Reliability predictor (authors' EEGNetLossPredictor) + its training objective.
# --------------------------------------------------------------------------
class EEGNetLossPredictor(nn.Module):
    """3-layer MLP on the flattened backbone features -> a scalar predicted loss
    (authors' models/losspredictor.py). ``in_features`` = backbone.out_features
    = F2 * (eeg_length // 32)."""

    def __init__(self, in_features: int):
        super().__init__()
        h1 = max(in_features // 2, 1)
        h2 = max(in_features // 4, 1)
        self.net = nn.Sequential(
            nn.Linear(in_features, h1, bias=True), nn.ELU(),
            nn.Linear(h1, h2, bias=True), nn.ELU(),
            nn.Linear(h2, 1, bias=True),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)                              # (B, 1)


def _soft_rank(x: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Differentiable rank of a 1-D vector via pairwise-sigmoid (soft-rank)."""
    d = x.unsqueeze(1) - x.unsqueeze(0)                     # (K, K)
    return torch.sigmoid(d / tau).sum(dim=1)               # (K,)


def _soft_spearman_loss(pred: torch.Tensor, target: torch.Tensor,
                        tau: float = 0.05) -> torch.Tensor:
    """1 - soft Spearman rank correlation between ``pred`` and ``target`` (both
    length-K vectors). Differentiable surrogate for the release's SoDeep sorter."""
    rp = _soft_rank(pred, tau)
    rt = _soft_rank(target, tau)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    corr = (rp * rt).sum() / (rp.norm() * rt.norm() + 1e-8)
    return 1.0 - corr


class BFT(Strategy):
    # Test-time adaptation: the model keeps refreshing its BN statistics on the
    # target stream, so it is a 'tta' strategy (the Exp streams the raw target and
    # BFT aligns it online, exactly like Tent / T-TIME).
    mode = "tta"
    K = 12

    def fit(self, model: nn.Module, source: EEGEpochs, ctx: RunContext) -> nn.Module:
        """Build and train BFT's own (t-1)-second EEGNet + reliability predictor.

        The Exp passes its full-length pipeline ``model`` but ignores ``fit``'s
        return value (it re-uses the same object for ``predict``). BFT needs a
        *truncated-input* model, so it builds its own and stashes it on the passed
        ``model`` for ``predict`` to pick up. The source arrives EA-aligned."""
        from hustbciml.core import registry
        from hustbciml.core.pipeline import PipelineModel

        cfg, device = ctx.cfg, ctx.device
        sfreq = float(source.sfreq)
        T = source.n_times
        eeg_length = _eeg_length(T, sfreq)
        if eeg_length == T:
            ctx.log(f"  [BFT] trial too short to truncate (T={T}, sfreq={sfreq}); "
                    f"sliding-window views degrade to identity.")

        # --- 1) BFT's own (t-1)-second decoder, benchmark capacity (F1/D/F2) ----
        backbone = registry.build(
            "models", cfg.backbone,
            n_chans=source.n_channels, n_times=eeg_length,
            n_classes=cfg.n_classes, sfreq=sfreq,
            F1=cfg.F1, D=cfg.D, F2=cfg.F2, dropout=cfg.dropout,
        )
        head = registry.build(
            "heads", cfg.head, in_features=backbone.out_features, n_classes=cfg.n_classes,
        )
        inner = PipelineModel(backbone, head).to(device)

        # Train BFT's (t-1)-second decoder with the paper's on-the-fly augmentation
        # (Sec. V-C): each source trial is randomly replaced by ONE of ~14
        # knowledge-guided transforms per epoch. Channel Reflection is label-aware
        # and left/right-hand only, so it enters the menu solely for L/R datasets
        # (dropped on right-hand-vs-feet, where the mirror has no valid label swap).
        inner = _bft_train(inner, source, ctx, sfreq, eeg_length)
        # surface the held-out-source val score on the passed model for the tuner
        setattr(model, "_val_score", getattr(inner, "_val_score", None))

        # --- 2) reliability predictor: rank-match softmax(-pred) to softmax(-CE) --
        temp = float(cfg.hp.get("bft_temp", 0.25))
        lp_epochs = int(cfg.hp.get("bft_lp_epochs", 20))
        lp = EEGNetLossPredictor(int(backbone.out_features)).to(device)
        opt = torch.optim.Adam(lp.parameters(), lr=cfg.lr)
        ce = nn.CrossEntropyLoss()

        Xf = source.X.astype(np.float64)            # FULL length (sliding windows crop from it)
        yf = source.y.astype(np.int64)
        N, C, _ = Xf.shape
        bs = max(2, cfg.batch_size)
        inner.eval()                                # frozen decoder; only lp trains
        for p in inner.parameters():
            p.requires_grad_(False)
        for epoch in range(lp_epochs):
            perm = np.random.RandomState(cfg.seed + epoch).permutation(N)
            lp.train()
            for s in range(0, N - bs + 1, bs):
                idx = perm[s:s + bs]
                B = len(idx)
                yb = torch.from_numpy(yf[idx]).to(device)
                views = _bft_views(Xf[idx], sfreq, eeg_length, cfg.seed + epoch)  # 12 x (B,C,L)
                K = len(views)
                allx = torch.from_numpy(
                    np.concatenate(views, 0).reshape(K * B, 1, C, eeg_length)).float().to(device)
                with torch.no_grad():
                    feats_all, logits_all = inner(allx)              # (K*B, feat), (K*B, n_cls)
                real = torch.stack([ce(logits_all.view(K, B, -1)[k], yb) for k in range(K)])  # (K,)
                pred = lp(feats_all).view(K, B).mean(dim=1)          # (K,) predicted loss per view
                loss = _soft_spearman_loss(F.softmax(-pred, dim=0), F.softmax(-real, dim=0))
                opt.zero_grad()
                loss.backward()
                opt.step()
        for p in inner.parameters():                # restore for BN-stat refresh at test
            p.requires_grad_(True)
        lp.eval()

        # stash on the passed model (predict reads these; fit's return is ignored)
        model._bft_inner = inner
        model._bft_lp = lp
        model._bft_temp = temp
        model._bft_eeg_length = eeg_length
        return model

    def predict(self, model: nn.Module, target: EEGEpochs, ctx: RunContext) -> Tuple[np.ndarray, np.ndarray]:
        cfg, device = ctx.cfg, ctx.device
        inner = model._bft_inner
        lp = model._bft_lp
        temp = float(getattr(model, "_bft_temp", cfg.hp.get("bft_temp", 0.25)))
        L = int(getattr(model, "_bft_eeg_length", _eeg_length(target.n_times, target.sfreq)))
        do_align = cfg.aligner != "Identity"
        sfreq = float(target.sfreq)
        raw = target.X.astype(np.float64)                   # (N, C, T) full, chronological
        N, C, T = raw.shape
        tb = int(cfg.test_batch)
        # Ablation: with bft_no_tta the trained truncated+CR-augmented decoder is
        # evaluated with online EA but WITHOUT the backprop-free test phase (no
        # K=12 views, no reliability weighting, no BN refresh) — BFT's own matched
        # base. The base-vs-full delta isolates the test-time contribution, the way
        # the paper's Table III does (EEGNet-with-aug -> BFT-A). Default off.
        no_tta = bool(cfg.hp.get("bft_no_tta", False))

        y_score = []
        run_rel = None                                      # running mean of per-view reliability
        n_seen = 0
        R, W = 0, None
        aligned_trunc = np.zeros((N, C, L), dtype=np.float64)   # (t-1)-s identity, for BN refresh
        for i in range(N):
            # ---- online Euclidean alignment of the current (full) trial ----
            if do_align:
                R = EA.online_update(raw[i], R, i)
                W = np.real(EA.inv_sqrt(R))
                cur = (W @ raw[i])[None]                     # (1, C, T)
            else:
                cur = raw[i][None]
            aligned_trunc[i] = cur[0, :, :L]

            if no_tta:                                       # matched-base ablation
                inner.eval()
                with torch.no_grad():
                    xb = torch.from_numpy(aligned_trunc[i][None, None]).float().to(device)  # (1,1,C,L)
                    _, logits = inner(xb)
                    y_score.append(torch.softmax(logits[0], dim=0).cpu().numpy())
                continue

            views = _bft_views(cur, sfreq, L, cfg.seed + i)  # 12 views of this trial, each (1,C,L)
            inner.eval()
            with torch.no_grad():
                xb = torch.from_numpy(
                    np.concatenate(views, 0).reshape(len(views), 1, C, L)).float().to(device)
                feats, logits = inner(xb)                    # (K, feat), (K, n_cls)
                view_probs = torch.softmax(logits / temp, dim=1)         # (K, n_cls)
                pred_losses = lp(feats).squeeze(1)           # (K,)
                rel = torch.softmax(-pred_losses, dim=0)     # per-trial reliability over K views
                # running mean of reliability across the stream (authors' design)
                run_rel = rel if run_rel is None else (run_rel * n_seen + rel) / (n_seen + 1)
                n_seen += 1
                w = run_rel / run_rel.sum()
                agg = (view_probs * w.unsqueeze(1)).sum(dim=0)   # (n_cls,) reliability-weighted
                y_score.append(agg.cpu().numpy())

            # ---- refresh BN running statistics on the recent (t-1)-s window ----
            if (i + 1) >= tb:
                batch = aligned_trunc[i - tb + 1: i + 1].reshape(tb, 1, C, L)
                xb = torch.from_numpy(batch).float().to(device)
                inner.train()                                # BN tracks the target batch
                with torch.no_grad():
                    inner(xb)
                inner.eval()

        y_score = np.asarray(y_score)
        return y_score.argmax(1), y_score
