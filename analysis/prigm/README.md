# Retrospective energy analysis

The [summary](energy_analysis_results.json) and [CSV reductions](reductions/) contain the retained energy-score and covariance comparisons for all 84 DART fits. They can be inspected directly. Recomputing them requires saved arrays and receipts that are not included in this public tree.

For each scoring geometry, the energy score is the mean distance from generated samples to observations minus half the mean distance between distinct generated samples. The variance analysis changes observation variance while keeping each encoder and decoder the same, then calculates the change in each term. The combined score is proper for each transformed distribution with finite first moments; interpreting the separate distance terms as calibration measures would require further evidence.

## Files and interpretation

| Files | Contents |
| --- | --- |
| `energy_terms_*.csv` | Both distance terms and their combined score, by fit and sampling stream |
| `reset_contrasts_*.csv` | Changes after estimating observation variance, using the same sampling streams |
| `skip_contrasts_*.csv` | Comparisons at the same original observation variance |
| `PPCA_*.csv` | PPCA baseline scores and neural comparisons |
| `covariance_structure.csv` | Empirical covariance fractions in the leading sixteen directions |

The scripts call the distance terms `A` and `B`; the paper uses `D_T` and `S_T` to distinguish them from model conditions. `B` is half the mean distance between distinct model samples. Changes are computed within sampling stream before averaging within each training seed. Separate medians of the two terms need not subtract to the median score.

PPCA and neural samples use separate streams, so their comparison uses stream means without pairing streams. Covariance fractions use the leading sixteen directions of weighted training covariance. These retrospective summaries do not establish the cause of training differences, Gaussianity or population confidence intervals.

## Recompute with retained inputs

The calculation needs the original evidence manifest with SHA256 `db618b2d01b7cd7263058c4e554f957e61fd0580d7663a608e787f45b261a42c`, the original core prepared arrays, and the extension prepared arrays plus all four diagnostic bundles. The original manifest and those complete inputs are not bundled. The current paper manifest cannot replace the original pinned manifest.

With these inputs available, run from the repository root, substituting their actual locations and fresh output paths:

```bash
python analysis/prigm/energy_two_term_analysis.py \
  --manifest /path/to/original-evidence.json \
  --core /path/to/retained-core --dart-work /path/to/retained-dart \
  --output runs/energy-analysis
python analysis/prigm/summarize_energy_analysis.py \
  --input-root runs/energy-analysis --output runs/energy-summary.json
```

The original CPU execution used Python 3.11.8, NumPy 1.26.4 and SciPy 1.14.1. The analysis authenticates consumed inputs and checks recomputed scores. The summary script additionally needs `output_manifest.json`, `verification.json`, `protocol.json` and `input_receipts.json` from that execution; the public CSV directory alone is insufficient. Protocol and input receipts contain local paths and should remain with the private run records.

The [analysis manifest](manifest.json) hashes these public files separately from their original evidence input. A documentation change updates its README entry without changing the scientific reductions or their producer identities.
