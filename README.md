# TruthRide: Simulation Code

This repository contains the simulation code and plotting scripts for the paper *TruthRide: A Trustworthy Closed-Loop Framework for Quality-Aware Recruitment and Retention in Mobile Crowdsensing*.

## Requirements

- Python 3.9 or later
- NumPy, Matplotlib, and Pillow

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install numpy matplotlib pillow
```

## Repository structure

- `dataset/`: raw mobility traces used to generate the simulation inputs
- `data/<dataset>/`: generated workers, tasks, and worker options
- `results/`: experiment outputs created at runtime; not tracked by Git
- `run_pipeline.py`: generates the simulation inputs for each dataset
- `run_revision_experiments.py`: runs the experiments reported in the revised manuscript
- `experiment2_core_*.py`: core algorithms, configuration, evaluation, and simulation components
- `experiment2_fig*.py`: scripts that reproduce individual figures
- `experiment2_*_sensitivity*.py`: parameter-sensitivity experiments and plots

## Datasets

Place the raw mobility traces in `dataset/` using these filenames:

| Dataset | File | Source |
|---|---|---|
| Beijing Taxi | `beijing_300_cars_2008-02-03.csv` | T-Drive trajectory data (Microsoft Research) |
| Rome Taxi | `crowd_temperature.csv` | Rome taxi traces with crowdsensed temperature readings |
| Chicago Bike | `Chicago_points.csv` | Divvy bike-share trip data (City of Chicago) |

The raw datasets are provided by third parties. Verify their licenses and redistribution terms before publishing them. If redistribution is not permitted, remove the files from `dataset/` and provide download instructions instead.

## Reproducing the results

Run all commands from the repository root.

```bash
# 1. Generate workers, tasks, and worker options in data/<dataset>/
python run_pipeline.py --datasets beijing rome chicago

# 2. Run all experiments from the revised manuscript
#    Outputs are written to results/<dataset>/ and results/summary_*.csv
python run_revision_experiments.py

# 3. Draw the figures added during revision
python experiment2_revision_figures.py

# 4. Draw an original per-round figure, for example for Chicago
cd results/chicago
python ../../experiment2_fig11_per_round_platform_utility.py
```

To run a single method:

```bash
TRUTHRIDE_DATASET=chicago python experiment2_core_main.py --mode lgsc --runs 10
```

On Windows Command Prompt, set the environment variable first:

```bat
set TRUTHRIDE_DATASET=chicago
python experiment2_core_main.py --mode lgsc --runs 10
```

## Methods

| `--mode` | Name in paper | Description |
|---|---|---|
| `random` | Random | Random recruitment under the budget |
| `epsilon_first` | Explore-First | Random exploration followed by mean-quality greedy selection |
| `cmab` | UCB-Greedy | Quality-aware CMAB recruitment module of TruthRide |
| `ucb_classic` | Classic CMAB-UCB | Standard UCB marginal-gain-per-cost greedy baseline |
| `trust` | Trust-Aware | `cmab` with spatial-consistency trust verification |
| `beta_rep` | Beta Reputation | `cmab` with Beta-reputation trust updates |
| `crh_td` | CRH Truth Discovery | `cmab` with CRH truth discovery and no trusted reference |
| `pgrd` | Membership-Aware | `trust` with the PGRD membership incentive |
| `lgsc` | TruthRide | Full framework: `pgrd` with LGSC sunk-cost retention |

## Information available to the platform (IEWV setting)

The recruitment and learning modules use only platform-observable information: worker bids, reported data `x_ij`, and the trust and category states maintained by the platform. Quality is estimated from consistency between reports:

```text
q_tilde_ij = 1 - min(1, abs(x_ij - ref_j) / sigma0)
```

Here, `ref_j` is the median of other trusted reports on the same task, or of other non-malicious peers. The ground-truth value `v_j` and simulated sensing quality `q_ij` are used only for data generation and evaluation. Set `QUALITY_FEEDBACK = "oracle"` in `experiment2_core_config.py` to reproduce the oracle upper bound.

## Main configuration switches

The following options are defined in `experiment2_core_config.py`:

| Key | Default | Meaning |
|---|---|---|
| `QUALITY_FEEDBACK` | `observable` | `observable` (IEWV) or `oracle` (upper bound) |
| `QUALITY_ERROR_SCALE` | `0.5` | Error scale `sigma0` for observable quality |
| `SELECTION_VALUE_FUNCTION` | `linear` | Recruitment value function: `linear`, `sigmoid`, or `hard` |
| `VALUE_FUNCTION` | `sigmoid` | Platform quality-value function used for evaluation |
| `SEED_TRUSTED_FRACTION` | `1.0` | Fraction of truly trusted workers initially known to the platform |
| `AVAILABILITY_KEEP_PROB` | `1.0` | Probability of retaining each available slot in the mobility simulation |

## Reproducibility notes

The scripts use fixed random seeds where configured, but exact floating-point results and figure rendering can vary slightly across Python and library versions. Record your Python version and installed package versions when archiving the final experimental results.

## License

The source code is released under the MIT License. See `LICENSE` for details. Third-party datasets remain subject to their respective licenses and redistribution terms.
