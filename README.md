# CyAnno Module

## What this module does

Provides the CyAnno benchmark module wiring and Python entrypoint.

- Entrypoint: `entrypoint_cyanno.py`
- Pipeline package: `cyanno_pipeline/`
- Module wrapper metadata: `module/`

This module is intended to run through Omnibenchmark module execution rather
than a dedicated local `run_*.sh` helper.

## Run locally

Use the entrypoint directly with benchmark-style arguments:

```bash
python models/cyanno/entrypoint_cyanno.py --name cyanno --output_dir <output_dir> --data.train_matrix <train.matrix.tar.gz> --data.train_labels <train.labels.tar.gz> --data.test_matrix <test.matrices.tar.gz>
```

## Run as part of benchmark

Configured in `benchmark/Clustering_conda.yml` analysis stage and executed with:

```bash
just benchmark
```

## What is needed to run

- Python environment with CyAnno dependencies from the benchmark env
- Preprocessing outputs (train matrix, train labels, test matrices)
- Writable output directory for `*_predicted_labels.tar.gz`
