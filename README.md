# Probabilistic Forecasting of Norwegian Wind Power Generation Two Days Ahead

## Abstract

Accurate and well-calibrated probabilistic wind power forecasts are essential for power system operation, particularly for day-ahead planning. In this work,
we present a probabilistic approach for two-day-ahead wind power forecasting for Norwegian bidding zones, with hourly resolution for the target day. It extends the WindAI 2025 competition by a focus on uncertainty quantification.

We formulate the task as a snapshot-to-snapshot regression problem and train separate models for each bidding zone. During training, the models use observed weather data to learn a mapping from local, wind-park-specific weather conditions to aggregated wind power generation at the bidding zone level without temporal offsets.
For two-day-ahead forecasting, this learned mapping is applied to the numerical weather prediction (NWP) ensemble forecast for the 24-hour horizon, starting two days ahead. The models map these predicted weather conditions to corresponding power generation, yielding a probabilistic wind power forecast of aggregated wind power over the forecast horizon. The final model predictions combine the uncertainty of weather forecasts and the predictive model uncertainty.

We compare a Bayesian Neural Network (BNN) and a Quantile Regression Neural Network (QRNN) with non-crossing quantiles, and evaluate performance using Root Mean Squared Error (RMSE) and Continuous Ranked Probability Score (CRPS).

Both models achieve largely comparable performance, while the BNN provides better-calibrated predictive distributions and slightly improved overall RMSE and CRPS values at significantly higher computational cost.
Based on these findings, we propose a practical two-stage workflow. First, a QRNN is used as an efficient proxy for model development and tuning, followed by a BNN to produce the final probabilistic predictions.

## Models

We implement:

- BNN with Gamma likelihood
- QRNN for simulaneous quantile prediction (non-crossing)

## Project Structure

`configs/`

- Dataset and Model Configuration files

`data/`

- Datasets: obtained from https://archive.sigma2.no/dataset/windai2025

`notebooks/`

- BNN_analysis.ipynb: BNN specific analysis (sample experimentsm posterior distribution,...)
- evaluation.ipynb: evaluates model and plots
- plausibility_checks.ipynb: polar plot plausbility check
- representative_test_set.ipynb: new representative test set (flexible length, and without Storm Amy), not used in thesis

`scripts/`

- run_experiments.py: train models
- run_evaluation.py
- run_plots.py
- run_all_evaluations_plots.sh: runs model evaluations and plotting function for all models, weather dataset, splits and bidding zones

`src/`

- All source code for models, training, sampling, evaulating, plotting

## Datasets

### Weather data

Available for the wind park locations

`Nowcast`: best estimate of the current atmospheric state, not ahead of time

`Forecast`: predictions, ensemble members represent different future weather scenarios

### Wind power data

`Windpower`: observed generated wind power for each bidding zone (NO1-NO4)

`Meta data`: for mapping of wind parks to bidding zones, max. theoretic capacity per park,..

## Installation

```bash
git clone https://github.com/noemiwbr/master_thesis.git
conda env create -f environment.yml -n masters
conda activate masters
```

## Usage

### Training

Run training with optional experiment config paths:

```bash
cd master_thesis/scripts
python run_experiments.py [--exp_path PATH [PATH ...]] [--synthetic]
```

- `--exp_path` (optional): One or more paths to experiment config files
- `--synthetic` (optional): Use synthetic target data

Example:

```bash
cd master_thesis/scripts
python run_experiments.py \
  --exp_path "../configs/model/bnn_gamma_01.yaml" \
             "../configs/model/qrnn_01.yaml"
```

If no `--exp_path` is provided, default configurations will be used.

### Evaluation & Plotting

Run evaluation and generate plots for specified model, dataset, split and bidding zone.

```bash
python scripts/run_evaluation.py \
  --dataset DATASET \
  --split SPLIT \
  --model MODEL \
  --bidding_zone ZONE

python scripts/run_plots.py \
  --dataset DATASET \
  --split SPLIT \
  --model MODEL \
  --bidding_zone ZONE
```

**Required arguments:**

- `--dataset`: one of `forecast`, `nowcast`
- `--split`: one of `train`, `validation`, `test`
- `--model`: one of `QRNN`, `BNN`, `shifted`, `median`
- `--bidding_zone`: one of `NO1`, `NO2`, `NO3`, `NO4`

Example:

```bash
python scripts/run_evaluation.py \
  --dataset forecast \
  --split test \
  --model BNN \
  --bidding_zone NO1
```

Run evaluation and generate plots for all configurations at once:

```bash
chmod +x run_all_evaluations_plots.sh
./run_all_evaluations_plots.sh
```
