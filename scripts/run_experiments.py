import time
import csv
import traceback
from pathlib import Path
import torch
import jax
from numpyro.infer.autoguide import AutoDiagonalNormal
import sys

sys.path.append("../src")
from configs.load_config import load_config
from data_processing.load import load_dataset
from models.build import build_model
from training.train import train_torch, sample_numpyro, run_svi
from training.evaluate import evaluate_model, evaluate_numpyro
from jax import random
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENTS = [
    "../configs/model/torch_model_quantile.yaml",
    # "../configs/model/bnn_gen_model_nowcast_01.yaml",
    # "../configs/model/bnn_gen_model_nowcast_02.yaml",
    # "../configs/model/bnn_gen_model_nowcast_03.yaml",
    # "../configs/model/bnn_gen_model_nowcast_04.yaml",
    # "../configs/model/bnn_gen_model_nowcast_03_emb.yaml",
    # "../configs/model/bnn_gen_model_nowcast_03_emb.yaml",
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


def run_single_experiment(exp_path: str) -> dict:
    start_time = time.time()

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

    # ---- model ----
    model = build_model(model_cfg, data)

    if model_cfg.backend == "torch":
        # ---- training ----
        if model_cfg.params.quantiles is None:
            quantiles = torch.linspace(0, 1, 51)
        else:
            quantiles = torch.tensor(model_cfg.params.quantiles)

        model, losses = train_torch(
            model=model,
            data=data,
            quantiles=quantiles,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
        )
        runtime = time.time() - start_time

        # ---- evaluation ----
        metrics, _ = evaluate_model(
            model=model,
            data=data,
            quantiles=quantiles,
            model_cfg=model_cfg,
        )

    elif model_cfg.backend == "numpyro":
        rng_key, rng_key_pred = random.split(random.PRNGKey(0))
        if model_cfg.sampler == "vi":
            # ---- optimize ----
            rng_key, rng_key_svi = random.split(rng_key)
            guide = AutoDiagonalNormal(model)  # TODO: better guide?
            svi_result = run_svi(
                model,
                data["X_train"],
                data["Y_train_gen"],
                rng_key_svi,
                guide,
            )

            runtime = time.time() - start_time

            # ---- evaluation ----
            metrics, _, _, _, _, _ = evaluate_numpyro(
                model=model,
                rng_key_predict=rng_key_pred,
                data=data,
                model_cfg=model_cfg,
                svi_params=svi_result.params,
            )

        elif model_cfg.sampler == "mcmc":
            # ---- sample ----
            samples, _, _ = sample_numpyro(model, data, model_cfg, data_cfg, rng_key)
            runtime = time.time() - start_time

            # ---- evaluation ----
            metrics, _, _, _, _, _ = evaluate_numpyro(
                model=model,
                rng_key_predict=rng_key_pred,
                data=data,
                model_cfg=model_cfg,
                samples=samples,
            )

    # ---- result row ----
    result = {
        "experiment": experiment_name,
        "status": "success",
        "rmse_median": float(metrics["rmse_median"]),
        "crps": float(metrics["crps"]),
        "coverage_025_975": float(metrics["coverage_025_975"]),
        "coverage_0_50": float(metrics["coverage_0_50"]),
        "runtime_sec": round(runtime, 2),
        "loss": losses[-1] if model_cfg.backend == "torch" else None,
        "error": "",
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
    args = parser.parse_args()

    for exp_path in args.exp_path:
        print("=" * 80)
        print(f"Starting: {exp_path}")

        try:
            result = run_single_experiment(exp_path)

            print("Finished successfully")
            print(f"  RMSE (median): {result['rmse_median']:.3f}")
            print(f"  CRPS: {result['crps']:.3f}")
            print(f"  Coverage 2.5–97.5%: {result['coverage_025_975']:.3f}")
            print(f"  Coverage 0–50%: {result['coverage_0_50']:.3f}")
            print(f"  Runtime: {result['runtime_sec']} s")

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
            }

        append_to_summary(result)

    print("\nAll experiments finished.")
    print(f"Summary written to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
