# hustbciml documentation

`hustbciml` is a unified framework for EEG decoding whose value is
**algorithm coverage**: many of the lab's methods ported onto one stage
architecture and measured, head to head, under a controlled comparison.

An **algorithm** is a named composition of five plug-in stages:

```
Aligner → Augmenter → Backbone → Head , driven by a Strategy
```

Every controlled comparison holds four of those stages at the canonical
configuration (`EA · no-aug · EEGNet · Linear · ERM`) and varies exactly one, so
each accuracy difference is attributable to a single change. All numbers are a
3-seed mean ± the standard deviation across seeds on **BNCI2014001**,
cross-subject leave-one-subject-out (9 subjects, 2-class, chance 50%).

## Read in this order

1. **[Glossary](glossary.md)** — the unified terms. Data hierarchy, tensor shapes, the five stages, the protocol split rule, the metrics. Start here if a word is unclear.
2. **[Results](../RESULTS.md)** — the controlled-comparison leaderboard: one table per stage axis, each row a single-axis change with its Δacc, plus the multi-seed ensemble.
3. **[Algorithm cards](cards/README.md)** — one card per method: mechanism, exact stage configuration, measured accuracy vs its reference range, paper citation, and vendored-code / license posture. Generated from the reproduction registry, so the numbers cannot drift from the results.
4. **[Porting guide](porting_guide.md)** — how to add a method: the seven steps, the controlled-comparison rule, and the measurement-integrity requirements (reproduce the range, keep faithful negatives, disclose every deviation).

## Where things live

| What | Path |
|---|---|
| Architecture overview, run commands | [`../README.md`](../README.md) |
| Stage contracts (the five ABCs) | `../core/stages.py` |
| Data containers (`EEGEpochs`, `EEGBatch`) | `../core/batch.py` |
| Numeric registry (measured numbers, citations, notes) | `../tests/repro/repro_targets.yaml` |
| Card prose (mechanism, license) | `cards/_content.yaml` |
| Card generator | `../scripts/build_cards.py` |
| Results aggregator (tables + Δacc) | `../scripts/compare.py` |

> Status: actively developed. Released under the MIT License (see [`../LICENSE`](../LICENSE)).
