# shapes.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Side-effect-free construction-time shape probing for backbones.

Most backbones here are sized from the data rather than hard-coded: they build
their layers, push a dummy tensor through, and read ``out_features`` (or a token
count, or a sequence length) off the result. That is the trick that lets one
config run on any dataset.

The probe must leave no trace, and a bare ``with torch.no_grad():`` does not
achieve that. ``no_grad`` suppresses gradients; it does **not** switch the module
to eval mode. So during the probe:

* every BatchNorm layer folds the dummy tensor's statistics into its running
  estimates — a tensor of zeros, which drags ``running_var`` away from 1 and
  bumps ``num_batches_tracked`` before a single real batch is seen;
* every Dropout layer draws from the global RNG, advancing the random stream by
  an amount that depends on the architecture.

Neither is catastrophic on its own — BN running stats are overwritten by real
batches during training, and a shifted RNG stream is just a different random
draw. But the benchmark's premise is that two rows differ in exactly one stage,
and two backbones compared at the same seed were starting from different random
states and different BN buffers purely because they probe differently. 13 of the
19 shipped backbones probed; EEGNet, which nearly every non-network row uses, did
not — so the network table was the one carrying the inconsistency.

``probe`` fixes it: eval mode for the duration, then the buffers and the RNG
state are restored exactly as they were.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def probe(module: torch.nn.Module):
    """Run a construction-time dummy forward that leaves no trace.

    Usage inside a backbone's ``__init__``, after the layers exist::

        with probe(self):
            self.out_features = self._feat(torch.zeros(1, 1, n_chans, n_times)).shape[1]

    Restores training mode, every buffer (BatchNorm running statistics and
    counters included), and the CPU RNG state. Construction happens on CPU — the
    pipeline moves the model to its device afterwards — so the CPU generator is
    the only one involved.
    """
    was_training = module.training
    saved = [(b, b.detach().clone()) for b in module.buffers()]
    rng_state = torch.random.get_rng_state()
    module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        module.train(was_training)
        with torch.no_grad():
            for buf, snapshot in saved:
                buf.copy_(snapshot)
        torch.random.set_rng_state(rng_state)
