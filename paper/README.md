# Paper and reproduction

Run commands from the repository root. The current manuscript is [prigm/main.tex](prigm/main.tex), with its retained [PDF](prigm/main.pdf) and [bibliography](prigm/references.bib).

| File | Role |
| --- | --- |
| [paper_evidence_manifest.json](prigm/paper_evidence_manifest.json) | Scientific inputs, numerical results and original output hashes |
| [presentation_manifest.json](prigm/presentation_manifest.json) | Seven replacement figure PDFs and their plotting inputs |
| [submission_manifest.json](prigm/submission_manifest.json) | Identities recorded for one compiled PDF and its sources |

## Verify a build

Install the root [CPU dependencies](../requirements-release.txt), LaTeX with `latexmk`, pdfLaTeX and BibTeX, and Poppler's `pdfinfo`, `pdftotext` and `pdffonts`.

```bash
latexmk -g -cd -pdf -recorder -interaction=nonstopmode -halt-on-error \
  paper/prigm/main.tex

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python \
  src/verify_prigm_release.py --paper-dir paper/prigm --camera-ready \
  --expected-core-sha256 37cc7445600eee541ee50aa580f6fdd6c0f37cf420074dbecc635c8d202946f5 \
  --expected-combined-sha256 1a20c43a29e640ff590e4c0f553bc7f08f2fad46fe6e2dab0e18806e20104fee \
  --expected-presentation-sha256 fdd0e5d582efae582f9aba34a1fdd461c7d520399ad61dd0581103d116184c1e
```

The forced build avoids relying on checkout timestamps. Verification needs the generated `main.bbl`, `main.log` and `main.fls`. It checks source and asset hashes, evidence identities, page limits, fonts, references and compiler inputs. It does not establish scientific validity or submission acceptance.

Rebuilt PDF bytes can differ. The verifier checks the current build; it does not compare it with the saved submission manifest. To replace a release, inspect the rebuilt pages, repeat verification with `--write-manifest`, and retain the resulting PDF and manifest together. The three expected hashes above identify the retained evidence and presentation; a text-only edit does not change them.

Keep the scientific source files, original PNGs and generated table fragments even when the manuscript includes replacement PDFs. The verifier authenticates the complete scientific output set, and the figure loader authenticates all thirteen scientific source files. Relative paths are part of these contracts.

## Train a new model from the original experiment

Training uses a separate environment from the CPU review requirements. The recorded stack used Python 3.11.8, TensorFlow 2.15.0, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3 and h5py 3.12.1:

```bash
python3.11 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install tensorflow==2.15.0 numpy==1.26.4 scipy==1.14.1 \
  pandas==2.2.3 h5py==3.12.1
```

This is a record of direct package versions, not a lock for GPU drivers or all transitive dependencies. With a compatible GPU environment, prepare a fresh smoke run and train one Gaussian condition:

```bash
python src/prigm_experiment.py prepare \
  --run-root runs/core-smoke --purpose smoke
python src/prigm_experiment.py train --run-root runs/core-smoke \
  --dataset gaussian --arm A --seed 42 --gpu-uuid 'GPU-REPLACE-ME'
```

Replace the UUID with an available physical GPU assigned to the run. Preparation itself runs on CPU. The smoke protocol uses a small number of steps to check execution; it does not reproduce a paper result.

For a new 60-fit core campaign, prepare a fresh directory without `--purpose smoke`, then use four distinct available GPU UUIDs:

```bash
python src/prigm_experiment.py prepare --run-root runs/core-new
python src/prigm_experiment.py campaign --run-root runs/core-new \
  --gpu-uuids 'GPU-UUID-0' 'GPU-UUID-1' 'GPU-UUID-2' 'GPU-UUID-3'
```

Interrupted fits restart in new attempts; optimizer and random-number-generator state are not resumed exactly. New runs have their own source and output identities.

## Full-study inputs

Replaying all 96 fits or rebuilding the complete scientific evidence requires retained core/refinement archives, checkpoint components, numerical controls, extension data and preparation/admission records, and saved diagnostic samples. These inputs and historical launchers are not bundled. The original refinement procedure also requires its original producer source; a new core run is not a substitute.

The source includes those data/model/evaluation routines for inspection. Public figure rendering uses the compact cache described in the [figure guide](../analysis/prigm/figures/README.md). The [retrospective analysis guide](../analysis/prigm/README.md) identifies its separate required inputs.

## License boundaries

The root [MIT notice](../LICENSE) covers repository software and associated documentation. The NeurIPS style retains its author attribution. NOAA data and cited works retain their source attribution. No separate manuscript license is declared here.
