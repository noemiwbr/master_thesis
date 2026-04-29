import time
import csv
import traceback
from pathlib import Path
import torch
import jax
import sys

sys.path.append("../src")
from configs.load_config import load_config
from data_processing.load import load_dataset
from data_processing.synthetic_data import synthetic_power_data
from models.build import build_model
from training.train import train_torch, sample_numpyro
from training.evaluate import evaluate_model, evaluate_numpyro
from jax import random
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = [
    "../configs/model/torch_model_quantile_04.yaml",
    "../configs/model/bnn_gen_model_nowcast_01_weighted.yaml",
]

RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "summary.csv"

SUMMARY_FIELDS = [
    "experiment",
    "status",
    "rmse_median",
    "crps",
    "coverage_025_975",
    "coverage_0_50",
    "runtime_sec",
    "error",
    "loss",
    "checkpoint",
]


# ============================================================================
# UTILITIES
# ============================================================================


def init_summary_file():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()


def append_to_summary(row: dict):
    with open(SUMMARY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writerow(row)


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================


def run_single_experiment(exp_path: str, synthetic: bool) -> dict:
    if synthetic:
        print("Using synthetic target data")

    # ---- load configs ----
    model_cfg = load_config(exp_path)
    data_cfg = load_config(model_cfg.data.path)

    assert (
        data_cfg.bid_zone == model_cfg.data.bid_zone
    ), f"Bid zones have to align in model and data cfg, got {data_cfg.bid_zone} and {model_cfg.data.bid_zone}"

    experiment_name = model_cfg.experiment_name
    print(f"\nRunning experiment: {experiment_name}")

    # ---- data ----
    data = load_dataset(data_cfg)

    if synthetic:
        data["Y_train_gen"], _ = synthetic_power_data(data, "X_train")
        data["Y_test_gen"], _ = synthetic_power_data(data, "X_test")

    # ---- model ----
    start_time = time.time()
    model = build_model(model_cfg, data)

    if model_cfg.backend == "torch":
        # ---- training ----
        if model_cfg.params.num_quantiles is not None:
            quantiles = torch.linspace(0, 1, model_cfg.params.num_quantiles)
        else:
            quantiles = torch.linspace(0, 1, 51)

        model, losses, checkpoint_name = train_torch(
            model=model,
            data=data,
            quantiles=quantiles,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
        )
        runtime = time.time() - start_time

        # ---- evaluation ----
        metrics, _, _ = evaluate_model(
            model=model,
            data=data,
            quantiles=quantiles,
            model_cfg=model_cfg,
        )

    elif model_cfg.backend == "numpyro":
        rng_key, rng_key_pred = random.split(random.PRNGKey(0))
        if model_cfg.sampler == "mcmc":
            # ---- sample ----
            samples, _, _, checkpoint_name = sample_numpyro(
                model, data, model_cfg, data_cfg, rng_key
            )
            runtime = time.time() - start_time

            # ---- evaluation ----
            metrics, _, _, _, _, _ = evaluate_numpyro(
                model=model,
                rng_key_predict=rng_key_pred,
                data=data,
                model_cfg=model_cfg,
                samples=samples,
            )
        else:
            raise ValueError(f"Unknown sampler: {model_cfg.sampler}")

    # ---- result row ----
    result = {
        "experiment": (
            experiment_name if not synthetic else experiment_name + " synthetic"
        ),
        "status": "success",
        "rmse_median": float(metrics["rmse_median"]),
        "crps": float(metrics["crps"]),
        "coverage_025_975": float(metrics["coverage_025_975"]),
        "coverage_0_50": float(metrics["coverage_0_50"]),
        "runtime_sec": round(runtime, 2),
        "loss": losses[-1] if model_cfg.backend == "torch" else None,
        "error": "",
        "checkpoint": checkpoint_name,
    }

    return result


# ============================================================================
# MAIN LOOP
# ============================================================================


def main():
    init_summary_file()
    jax.config.update("jax_num_cpu_devices", 6)  # I have 8 physical cores
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument(
        "--exp_path",
        nargs="*",
        type=str,
        default=EXPERIMENTS,
        help="config file paths for experiments",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic target",
    )

    args = parser.parse_args()

    for exp_path in args.exp_path:
        print("=" * 80)
        print(f"Starting: {exp_path}")

        try:
            result = run_single_experiment(exp_path, args.synthetic)

            print("\nFinished successfully")
            print(f"   Runtime: {result['runtime_sec']} s")
            print("\n Validation statistics:")
            print(f"   RMSE val: {result['rmse_median']:.3f}")
            print(f"   CRPS val: {result['crps']:.3f}")
            print(f"   Coverage 2.5–97.5 val: {result['coverage_025_975']:.3f}")
            print(f"   Coverage 0–50 val: {result['coverage_0_50']:.3f}")

        except Exception as e:
            print("Experiment failed")
            traceback.print_exc()

            result = {
                "experiment": Path(exp_path).stem,
                "status": "failed",
                "rmse_median": "",
                "crps": "",
                "coverage_025_975": "",
                "coverage_0_50": "",
                "runtime_sec": "",
                "loss": "",
                "error": str(e),
                "checkpoint": "",
            }

        append_to_summary(result)

    print("\nAll experiments finished.")
    print(f"Summary written to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
