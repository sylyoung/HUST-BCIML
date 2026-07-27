# Building the web app data

The web app (`docs/`) is static. Its content is compiled from three YAML files
in `gallery/data/` into JavaScript data files in `docs/data/` by `build_site.py`.

## Sources (edit these)

- **`data/publications.yml`** — one entry per paper:
  `id, title, authors, year, venue, doi, topic, paradigm[], code_url, tldr`.
  Hand-curated. `topic` is one of the 9 research pillars; `paradigm` is a list of
  BCI paradigm tags (`MI`, `P300`, `SSVEP`, `Seizure`, `Affect`, `Drowsy`,
  `Speech`, `iBCI`, `Biometric`, `Sleep`).
- **`data/lab.yml`** — lab bio, links, the anchor project, and the flagship repos
  shown on the Overview page.
- **`data/benchmark.yml`** — the controlled-comparison leaderboard, one block per
  stage axis. Accuracy/kappa are synced from `hustbciml/RESULTS.md`. Each row's
  `key` is also the key of its provenance entry in
  `hustbciml/tests/repro/repro_targets.yaml` (paper citation, reference range,
  note) — but the two files are **separate hand-maintained sources**:
  `build_site.py` reads only the three files in this directory, and
  `hustbciml/scripts/build_cards.py` reads only `repro_targets.yaml` plus
  `docs/cards/_content.yaml`. Editing one does **not** update the other. That is
  why `hustbciml/tests/repro/test_repro_targets.py` exists: it fails the build if
  the two publish different numbers for the same method and dataset, or if a
  leaderboard key has no provenance entry. Update both in the same commit and let
  the test confirm it.

  Optional per-row honesty fields, both rendered on the site:
  `na_reason` (why a dataset cell is empty, so "n/a" is not read as "not run
  yet") and `also_varies` (what else the row changes besides its table's axis, so
  a Δ is not read as a single-stage effect).

## Generate

```bash
python3 gallery/build_site.py       # from the repo root; requires pyyaml
```

This writes:

- `docs/data/lab.js` — `window.LAB`, `window.SITE`
- `docs/data/publications.js` — `window.PUBLICATIONS`
- `docs/data/benchmark.js` — `window.BENCHMARK`

Each is a plain `window.X = <json>;` assignment. The data is inlined this way
(rather than fetched) so `docs/index.html` opens directly as a local file — no
server — and works unchanged on GitHub Pages.

## Preview

Open `docs/index.html` in a browser. No server required.
