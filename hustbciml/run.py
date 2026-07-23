# run.py  —  HUST-BCIML EEG-decoding benchmark.
# Author: Siyang Li <lsyyoungll@gmail.com>, 2026.
"""Entry point: ``python -m hustbciml.run --algorithm EA-EEGNet --dataset Toy``.

This is the command-line front door to the whole benchmark. It does four things
in order. It resolves the Config from the preset and CLI flags. It looks up the
protocol named in the Config in the ``PROTOCOLS`` table to get the Exp class
that runs it. It runs that Exp ``--itr`` times, each repeat on a seed one higher
than the last, so a reported number comes from several independent runs rather
than one lucky draw. Then it prints the mean and spread of the primary metric
across those repeats.

The ``--list`` flag short-circuits all of that and just prints the plug-in
catalog, which is the quickest way to see what aligners, backbones, and so on
are installed.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from hustbciml.core.config import resolve_config
from hustbciml.core.registry import catalog

# protocol registry — extend as protocols land
from hustbciml.exp.exp_cross_subject import Exp_CrossSubject

# Maps a protocol name (the ``--protocol`` value) to the Exp class that
# implements it. This is the data axis of the benchmark: each Exp defines how
# subjects and sessions are split into source and target. Adding a protocol
# means writing its Exp and adding one line here. The commented entries are the
# planned protocols not yet implemented.
PROTOCOLS = {
    "cross_subject": Exp_CrossSubject,
    # "within_subject": ...,  (M1)
    # "cross_session": ...,   (M1)
    # "online": ...,          (M1)
}


def _print_catalog():
    """Print every installed plug-in, grouped, for ``--list``."""
    print("Available plug-ins:")
    for group, names in catalog().items():
        print(f"  {group:11s}: {', '.join(names)}")


def main(argv=None):
    cfg, ns = resolve_config(argv)
    if ns.list:
        # ``--list`` is informational only: show the catalog and stop before any
        # data is loaded or any run starts.
        _print_catalog()
        return

    if cfg.protocol not in PROTOCOLS:
        raise SystemExit(
            f"protocol {cfg.protocol!r} not implemented yet; available: {sorted(PROTOCOLS)}"
        )
    Exp = PROTOCOLS[cfg.protocol]

    # Repeat the run ``itr`` times for a variance estimate. Each repeat gets its
    # own config copy with the seed bumped by the repeat index, so the base
    # ``cfg`` is left untouched and repeat i is fully reproducible from its seed.
    primaries = []
    for i in range(cfg.itr):
        cfg_i = dataclasses.replace(cfg, seed=cfg.seed + i)
        print(f"\n########## itr {i + 1}/{cfg.itr}  (seed {cfg_i.seed}) ##########")
        summary = Exp(cfg_i).run()
        # Collect the one headline number from each repeat. The Exp already
        # averages this over subjects within a single run.
        primaries.append(summary["primary"]["mean"])

    # Only report an aggregate line when there is more than one repeat to
    # aggregate. A single run has already printed its own result.
    if cfg.itr > 1:
        arr = np.array(primaries)
        print(f"\n######### {cfg.setting()} #########")
        print(f"primary over {cfg.itr} itr: {arr.mean():.2f} +/- {arr.std():.2f}")


if __name__ == "__main__":
    main()
