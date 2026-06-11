# TrajRACE Main Experiment Artifact

This repository provides the scripts for reproducing the main experiment of:

**Risk-Aware Road-Network Trajectory Synthesis under Local Differential Privacy**

The artifact includes the main TrajRACE pipeline: preprocessing, event construction, risk computation, local perturbation, statistical recovery, synthetic trajectory generation, and evaluation.

## Environment

We recommend Python 3.9.

```bash
conda create -n trajrace python=3.9
conda activate trajrace
pip install -r requirements.txt
```

## Data

This minimal artifact uses Porto as an example dataset. Raw Porto data are not included due to data license restrictions.

Please download the Porto taxi trajectory dataset and place it under:

```text
data/raw/porto/
```

The output files will be generated under `data/` and `outputs/`.

## Run the Main Pipeline

Run the following scripts in order:

```bash
python scripts/preprocess_dataset.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/build_events.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/compute_transition_risk.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/privatize_riskaware.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/privatize_uniform.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/recover_statistics.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/synthesize_trajectories.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml

python scripts/evaluate_compare.py --dataset_config configs/dataset_porto_100k.yaml --exp_config configs/exp_main_100k.yaml
```

## Pipeline Description

* `preprocess_dataset.py`: preprocesses raw trajectories and converts them into road-segment sequences.
* `build_events.py`: extracts start, segment-count, and transition events.
* `compute_transition_risk.py`: computes transition-level risk scores and protection buckets.
* `privatize_riskaware.py`: applies the proposed risk-aware local perturbation mechanism.
* `privatize_uniform.py`: applies the uniform-protection variant for comparison.
* `recover_statistics.py`: recovers start, segment-count, and conditional transition statistics.
* `synthesize_trajectories.py`: generates synthetic road-network trajectories.
* `evaluate_compare.py`: evaluates recovered statistics and synthetic trajectories.

## Outputs

The main outputs include:

```text
outputs/tables/
outputs/reports/
outputs/figures/
```

The evaluation script reports diagnostic metrics such as high-risk keep rate, transition JS divergence, bigram JS divergence, and transition support ratio.

## Notes

This repository contains only the main experimental pipeline. Raw datasets, large intermediate files, logs, and full baseline implementations are not included.

Please make sure that all paths in the YAML files are relative paths before running the scripts.
