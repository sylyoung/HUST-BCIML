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
  stage axis. Accuracy/kappa are synced from `hustbciml/RESULTS.md`; each row's
  `key` links to its provenance entry in `hustbciml/tests/repro/repro_targets.yaml`
  (paper citation, reference range, note), which the generator reads for the
  per-method cards.

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
