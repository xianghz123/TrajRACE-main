TrajRACE Main Experiment Artifact
This repository provides the main experimental pipeline for:
Risk-Aware Road-Network Trajectory Synthesis under Local Differential Privacy
TrajRACE processes road-network trajectories on the client side, applies risk-aware local perturbation, recovers population-level mobility statistics on the server, and generates road-network-consistent synthetic trajectories.
Environment
We recommend Python 3.10.
```bash
conda create -n trajrace python=3.10 -y
conda activate trajrace
pip install -r requirements.txt
```
Data
This artifact uses the Porto taxi trajectory dataset as the example dataset.
Raw Porto data are not included in this repository due to dataset licensing and file-size restrictions. Please place the Porto CSV file at:
```text
data/raw/porto/train.csv
```
The fixed public road graph is expected at:
```text
data/intermediate/road_network/porto.graphml
```
The main configuration files are:
```text
configs/dataset.yaml
configs/exp_main.yaml
```
All generated data and experiment results are stored in versioned subdirectories under:
```text
data/variants/
```
Run the Main Pipeline
Run the following scripts in order from the repository root:
```bash
python scripts/preprocess_dataset.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/build_events.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/compute_transition_risk.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/privatize_riskaware.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/audit_privacy.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/recover_statistics.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/privatize_uniform.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/recover_uniform_statistics.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/synthesize_trajectories.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml

python scripts/evaluate_compare.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml
```
The complete pipeline can also be executed through:
```bash
python scripts/run_rebuttal_experiments.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml \
  --mode single
```
For a small smoke test:
```bash
python scripts/run_rebuttal_experiments.py \
  --dataset_config configs/dataset.yaml \
  --exp_config configs/exp_main.yaml \
  --mode single \
  --raw_sample_size_override 200
```
Pipeline Description
`preprocess_dataset.py`: preprocesses raw GPS trajectories, performs map matching, and produces complete road-segment trajectories.
`build_events.py`: decomposes long trajectories into SBSs and extracts start, exact segment-count, and transition events.
`compute_transition_risk.py`: computes transition-level endpoint, long-stay, and low-degree risks and assigns fixed risk buckets.
`privatize_riskaware.py`: applies the proposed risk-aware local perturbation mechanism.
`audit_privacy.py`: checks public domains, report schemas, budget feasibility, and the implemented privacy conditions.
`recover_statistics.py`: recovers start, segment-count, and conditional transition statistics for TrajRACE.
`privatize_uniform.py`: applies the uniform-allocation variant under the same reporting framework.
`recover_uniform_statistics.py`: recovers statistics from the uniform reports.
`synthesize_trajectories.py`: generates synthetic road-network trajectories from the recovered statistics.
`evaluate_compare.py`: compares risk-aware and uniform mechanisms and evaluates the generated trajectories.
`run_rebuttal_experiments.py`: provides a reproducible entry point for single runs and multi-budget/multi-seed rebuttal experiments.
Outputs
Outputs are stored under a versioned directory of the form:
```text
data/variants/<dataset_variant>/
```
Experiment-specific results are stored under:
```text
data/variants/<dataset_variant>/experiments/<exp_tag>/
```
The main output directories include:
```text
risk/
privatized/
recovered/
synthetic/
evaluation/
audits/
```
The evaluation results include metrics such as transition JS divergence, synthetic bigram JS divergence, exact segment-count preservation, public-road legality, and risk-aware versus uniform protection statistics.
Notes
The road network and legal successor sets are treated as public information.
For transition perturbation, the current road segment is public context and the true successor is the protected value.
The privacy audit should finish with `OVERALL STATUS = PASS` and `Hard failures = 0`.
`B` is used as the SBS adaptive reporting-schedule cap in the current artifact.
Files generated with `--save_debug` may contain private truth for local debugging and should not be distributed as server-visible reports.
The 200-trajectory smoke test is only for checking correctness and reproducibility; paper-scale results should use the full experimental settings.
Raw preprocessing and map matching can be time-consuming and are separate from the method-specific runtime reported after common preprocessing.
