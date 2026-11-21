# Residual anomaly analyzer

Searches real nuclear data for unexplained residuals after subtracting published mass models, for heavy and
superheavy nuclei (Z >= 82). Built for the active-mass paper in `../paper`; its Sec. II D fit is the special case
this code reproduces exactly before doing anything else.

## Run

```
python -I ra_ingest.py        # raw published tables -> cache/  (standard library only, isolated)
python residual_analyzer.py   # the analysis -> out/results.json, residuals.npz, point_z.json  (about 2 minutes)
python ra_figures.py          # out/fig1..3 .png
python ra_report.py           # out/REPORT.md, every number read from results.json
```

`python residual_analyzer.py --validate` only checks that the paper's limits are reproduced.

## Layout

| File | Job |
|---|---|
| `ra_ingest.py` | Parses each raw table, validates it, writes one normalized cache plus `cache/SOURCES.json` (SHA-256, reference, rows kept and rejected, error against AME2020). The only code that touches raw files. |
| `ra_stats.py` | Observable-agnostic statistics: correlated-noise model fitted by REML, generalized least squares with nuisance terms profiled out, template-bank z-scores, look-elsewhere simulation. |
| `ra_nuclear.py` | Nuclear sector: builds binding-energy and Q_alpha residuals per model, the generic template bank, the paper's self-energy template, and the data cut-off table. |
| `ra_atomic.py` | Atomic sector: no data ingested; computes how large the paper's force would be in an atom (DATA_GAP records with the precision that would be needed). |
| `residual_analyzer.py` | Runs every pass for every model and writes the results. |
| `ra_figures.py`, `ra_report.py` | Figures and the written report, from the results only. |
| `raw/<source>/` | Tables exactly as published, one source per folder. |

## Data (all real, none synthetic)

AME2020 `mass.mas20` and `rct1.mas20`; mass models WS (Ma et al. 2019), WS4, FRDM2012, HFB-24, KTUY05. References,
URLs and file hashes are in `cache/SOURCES.json` after ingest. A value a table does not give is absent from the
cache; AME values estimated from systematics (#) are flagged and never used as measurements.

## Adding things

- **A mass model:** put its table in `raw/<name>/`, add a reader and an entry to `MODELS` in `ra_ingest.py`, rerun.
  Everything downstream picks it up.
- **An observable:** add a branch to `ra_nuclear.dataset` (for example S_2n, already in the cache) returning
  `Z, N, y, se`. The statistics do not care what `y` is.
- **A sector:** a module that returns the same `Z, N, y, se` arrays (or its own coordinates and distance matrix)
  can reuse `ra_stats` unchanged.

## Reading the results

A feature in one model's residuals is usually that model's error. The test that matters is section 3 of the
report: a real effect is in the data, so every model must show it with the same sign.
