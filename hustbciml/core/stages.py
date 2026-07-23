# stages.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Plug-in abstract base classes — the kinds a method can plug in as.

An *algorithm* in hustbciml is a named composition of stage plug-ins:

    Aligner -> Augmenter -> Backbone -> Head , driven by a Strategy.

Read left to right that is also the data flow. The Aligner whitens the numpy
``EEGEpochs`` per subject, the Augmenter perturbs the torch ``EEGBatch`` at
train time, the Backbone turns each trial into a feature vector, and the Head
turns features into class logits. The Strategy is not a link in that chain. It
is the driver that owns the training and inference procedure and calls the
other four in the right order.

Each base class below is an ABC that fixes one such role. A concrete plug-in
subclasses exactly one of them and fills in the abstract methods. That is the
whole contract, so a new method is added by writing a new subclass, not by
editing this file or any dispatcher.

Two things make the plug-ins composable without glue code. First, they all
speak the shared data contracts from ``batch.py`` (``EEGEpochs`` in numpy,
``EEGBatch`` in torch), so any aligner fits in front of any backbone. Second,
each class carries declarative attributes such as ``requires_labels``,
``supports_online``, ``is_gradient``, and ``mode``. These are plain class
constants, not behavior. The pipeline builder and the protocol read them to
check a composition makes sense (for example, that a label-free aligner is used
where target labels are hidden) before a run starts.

Alongside those five stage roles there is one more plug-in kind that is *not*
part of the pipeline chain: the ``Combiner``. A combiner runs after several
trained models have each predicted on the same target trials, and fuses those
predictions into one consensus label per trial. It is the plug-in kind of the
ensemble-learning family, discovered by the same registry from
``algorithms/ensembles/`` but applied by the ensemble runner scripts rather than
by ``build_pipeline``.

The registry resolves each plug-in from ``algorithms/<group>/<Name>.py`` where
the file name equals the class name equals the CLI key. So the class name you
write here is also the string a user passes on the command line.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .batch import EEGEpochs, EEGBatch


class Aligner(ABC):
    """Numpy, per-domain signal alignment (e.g. Euclidean Alignment).

    Operates on ``EEGEpochs`` before any tensor is formed, so it is the first
    thing that touches the raw signal. ``fit`` estimates per-domain reference
    statistics on the *training* domains only. At test time a fresh aligner is
    fit on each target subject from its own trials (offline), or its reference
    is updated incrementally as trials arrive (online). Because the reference is
    per subject and built from trials rather than labels, this normalization
    stays leakage-safe even when the target labels are withheld.

    Contract: a subclass implements ``fit`` (estimate and cache the per-domain
    references, return self) and ``transform`` (apply them, return a new
    container via ``epochs.with_X``). ``fit`` must not mutate its input.
    """

    # Declarative contract, read by the pipeline builder, not by this class.
    requires_labels: bool = False   # True if the method needs labels to align (label-alignment style)
    supports_online: bool = False   # True if the reference can be updated per-sample in a stream

    @abstractmethod
    def fit(self, epochs: EEGEpochs) -> "Aligner":
        """Estimate and cache the per-domain reference statistics. Return self."""
        ...

    @abstractmethod
    def transform(self, epochs: EEGEpochs) -> EEGEpochs:
        """Apply the cached alignment and return a new ``EEGEpochs`` (same y, domain)."""
        ...

    def fit_transform(self, epochs: EEGEpochs) -> EEGEpochs:
        """Convenience for the common in-place case: fit on these trials, then
        transform them. Used where the same epochs are both the reference and
        the thing to align, such as aligning one target subject on its own."""
        return self.fit(epochs).transform(epochs)


class Augmenter(ABC):
    """Train-only batch transform (e.g. Channel Reflection, time-freq aug).

    Runs on a torch ``EEGBatch`` inside the training loop, after the batch is
    formed and moved to the device. Its job is to enlarge or perturb the
    training distribution. It is a plain callable, not an ``nn.Module``, because
    it holds no learned parameters.

    Contract: given a batch, return a batch. The default ``Identity`` augmenter
    returns its input untouched. ``train_only = True`` tells the driver to skip
    augmentation at validation and test time so scoring sees the real signal.
    """

    train_only: bool = True   # driver applies this only during training, never at eval

    @abstractmethod
    def __call__(self, batch: EEGBatch) -> EEGBatch:
        """Transform one training batch and return the (possibly new) batch."""
        ...


class Backbone(nn.Module, ABC):
    """Feature-extractor: the learned map from a trial to a feature vector.

    This is the trainable core of the pipeline. It takes the ``(B, 1, C, T)``
    batch tensor and produces a ``(B, out_features)`` embedding, which the Head
    then reads. A subclass must do two things in ``__init__``: build its layers
    from the data-derived dimensions (``n_chans``, ``n_times``, ``n_classes``,
    ``sfreq``) that the pipeline passes in, and set ``self.out_features`` to the
    embedding width it produces, because the Head is sized from that number.

    ``task_name`` names the output mode. A backbone that can emit more than one
    kind of representation dispatches on it, so a single class serves several
    heads.
    """

    task_name: str = "classification"   # output mode a multi-head backbone dispatches on

    def __init__(self):
        super().__init__()
        # Width of the feature vector this backbone emits. A subclass overwrites
        # this in its own __init__ once its layers are built, and the Head reads
        # it as its input size. Left 0 here as an unset sentinel.
        self.out_features: int = 0

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch to features: ``(B, 1, C, T) -> (B, out_features)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Default forward is just feature extraction. PipelineModel calls
        # forward_features and the head separately, so this exists mainly for
        # using a backbone on its own.
        return self.forward_features(x)


class Head(nn.Module, ABC):
    """Maps backbone features to class logits: ``(B, out_features) -> (B, n_classes)``.

    There are two families, and ``is_gradient`` says which one a subclass is.
    A gradient head such as ``Linear`` is a differentiable layer that trains
    jointly with the backbone under the strategy's loss. A classical head such
    as LDA or MDM sets ``is_gradient = False`` and instead fits itself on numpy
    features and predicts from them, outside the gradient loop. The strategy
    reads this flag to decide whether the head joins backpropagation or is fit
    separately on extracted features.
    """

    is_gradient: bool = True   # False for classical fit/predict heads (LDA, MDM)

    @abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Turn a feature batch into per-class logits ``(B, n_classes)``."""
        ...


class Strategy(ABC):
    """Owns the train / adapt / predict *procedure* (ERM, DANN, T-TIME ...).

    The strategy is the driver of a run. It receives the composed model plus a
    ``RunContext`` (config, device, the augmenter and aligner, an optional
    unlabeled target) and decides how training and prediction actually happen.

    It is deliberately separated from the Exp protocol along two axes.
    ``Exp_<Protocol>`` owns the *data axis*: which subjects or sessions are
    source versus target, and how they are split. The Strategy owns the
    *procedure axis*: the optimization and adaptation recipe. Because the two
    are orthogonal, one protocol pairs with any strategy, which is what makes
    the benchmark a grid rather than a fixed set of scripts.

    ``mode`` tells the protocol how to run the strategy, so the harness does not
    hard-code any one method's flow:
      'gradient' — standard source training then frozen inference.
      'tta'      — test-time adaptation: model keeps updating on the target stream.
      'fit'      — classical fit/predict, no gradient loop.
    """

    mode: str = "gradient"
    # Whether the strategy is transductive, meaning it looks at the (unlabeled)
    # target during fit. When True, the Exp fills ``ctx.target_unlabeled`` with
    # the aligned, label-masked target epochs and the strategy reads it there.
    # DANN and MEKT set this; plain ERM does not.
    uses_target: bool = False

    @abstractmethod
    def fit(self, model: nn.Module, source: EEGEpochs, ctx) -> nn.Module:
        """Train ``model`` on labeled source epochs. Returns the trained model.

        A transductive strategy (``uses_target = True``) additionally reads the
        unlabeled target from ``ctx.target_unlabeled`` here.
        """

    @abstractmethod
    def predict(self, model: nn.Module, target: EEGEpochs, ctx) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on the target epochs.

        Returns ``(y_pred, y_score)`` where ``y_pred`` is ``(N,)`` hard labels
        and ``y_score`` is ``(N, n_classes)`` per-class scores. A ``'tta'``
        strategy may keep adapting the model while it walks the target stream,
        so predict is where test-time adaptation happens for those methods.
        """


class Combiner(ABC):
    """Post-hoc black-box ensemble combiner: fuse K base models into one prediction.

    A combiner is not a pipeline stage. It runs *after* several trained models
    have each produced predictions on the same target trials, and fuses those
    predictions into one consensus label per trial. "Black-box" means it sees
    only the models' outputs, never their weights, the raw signal, or any
    ground-truth label. This is the decentralized / test-time-ensemble setting:
    independent models (K random seeds of one algorithm, or the per-source
    learners of a decentralized ensemble, or several heterogeneous architectures)
    each vote, and the combiner decides the consensus.

    Contract: ``combine`` takes ``scores`` of shape ``(K, N, C)`` — K base models,
    N trials, C class scores — and returns ``(N,)`` hard consensus labels. Every
    combiner in this benchmark aggregates only the models' HARD votes (each
    argmaxes the scores internally). There is deliberately no soft-score-averaging
    combiner, because a soft mean would hand one method an information advantage
    over the hard-label crowdsourcing aggregators it is compared against, so the
    comparison would no longer be apples-to-apples.

    Declarative attributes, read by the ensemble runners, not by this class:
      name          — the display / CLI key. It may contain characters a file name
                      cannot (``SML-OVR``, ``M-MSR``, ``Dawid-Skene``), so it is
                      kept separate from the class name the registry keys on.
      lab_proposed  — True for the lab's own combiners (SML-OVR, StackingNet).
      binary_only   — True if the method is defined only for two classes (binary
                      SML); the runner skips it on multi-class tasks.
    """

    name: str = ""
    lab_proposed: bool = False   # True for the lab's own combiners
    binary_only: bool = False    # True if valid only for 2-class tasks (binary SML)

    @abstractmethod
    def combine(self, scores: np.ndarray) -> np.ndarray:
        """Fuse ``(K, N, C)`` base-model scores into ``(N,)`` consensus labels."""
        ...

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        # The ensemble runners invoke a combiner like a plain function
        # (``COMBINERS[name](scores)``), so __call__ forwards to combine. This
        # keeps a combiner instance a drop-in for the old module-level functions.
        return self.combine(scores)


class VoteCombiner(Combiner):
    """A Combiner that needs only the models' HARD votes, not their soft scores.

    Most crowd-label aggregators (Dawid-Skene, PM, EBCC, ...) are defined over a
    table of discrete votes: one integer label per model per trial. This base
    class performs the single shared step for all of them — argmax the
    ``(K, N, C)`` scores into a ``(K, N)`` integer vote table — and hands that to
    ``aggregate``. A subclass implements ``aggregate`` and derives the class count
    internally from the votes (``votes.max() + 1``), exactly as the original
    vote-table implementations did, so nothing about their numerical behavior
    changes: ``combine`` here is precisely the old ``lambda s: fn(s.argmax(2))``.
    """

    @abstractmethod
    def aggregate(self, votes: np.ndarray) -> np.ndarray:
        """Fuse ``(K, N)`` integer hard votes into ``(N,)`` consensus labels."""
        ...

    def combine(self, scores: np.ndarray) -> np.ndarray:
        return self.aggregate(scores.argmax(axis=2))
