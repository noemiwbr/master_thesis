import argparse
import sys

sys.path.append("../src")

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import properscoring as ps

from configs.load_config import load_config

from data_processing.load import (
    load_dataset,
    load_windpower,
    load_evaluation_forecast_data,
    load_scaler_params,
)

from data_processing.utils import (
    load_torch_model,
    load_checkpoint,
)

from data_processing.synthetic_data import synthetic_power_data

from models.build import build_model

from training.evaluate import (
    test_QR,
    test_BNN,
    evaluate_model,
    evaluate_numpyro_on_train_set,
    evaluate_numpyro,
    test_BNN_new,
    test_QR_new,
    crps_from_quantiles,
)

from training.plot import (
    plot_qr_hdi_prediction_simple,
    plot_qr_prediction_ensemble,
    plot_bnn_gamma_combined,
    extract_quantile_curves,
    window_plot,
    agg_weights_plot,
    calibration_diagnostics,
    plot_qr_hdi_prediction_lines,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINTS = {
    "BNN": {
        "ELSPOT NO1": "model_mw_gamma_extended_2026-03-28_14-17-43_nowcast_ELSPOT_NO1",
        "ELSPOT NO2": "model_mw_gamma_extended_2026-03-28_18-30-32_nowcast_ELSPOT_NO2",
        "ELSPOT NO3": "model_mw_gamma_extended_2026-03-29_01-53-28_nowcast_ELSPOT_NO3",
        "ELSPOT NO4": "model_mw_gamma_extended_2026-03-29_07-02-26_nowcast_ELSPOT_NO4",
    },
    "QRNN": {
        "ELSPOT NO1": "WindPowerQuantile_weighted_2026-04-09_10-26-25_nowcast_gen_ELSPOT_NO1",
        "ELSPOT NO2": "WindPowerQuantile_weighted_2026-04-09_10-26-57_nowcast_gen_ELSPOT_NO2",
        "ELSPOT NO3": "WindPowerQuantile_weighted_2026-04-09_10-27-30_nowcast_gen_ELSPOT_NO3",
        "ELSPOT NO4": "WindPowerQuantile_weighted_2026-04-09_10-28-02_nowcast_gen_ELSPOT_NO4",
    },
}
PATH_MODEL_CFG = {
    "BNN": lambda i: f"../configs/model/bnn_gen_model_nowcast_0{i}_weighted.yaml",
    "QRNN": lambda i: f"../configs/model/torch_model_quantile_0{i}.yaml",
}

DATA_SPLIT = {
    "train": "train",
    "validation": "test",
    "test": "eval",
}

evaluation_dates = [
    "2025-09-22",
    "2025-09-23",
    "2025-09-24",
    "2025-09-25",
    "2025-09-26",
    "2025-09-29",
    "2025-09-30",
    "2025-10-01",
    "2025-10-02",
    "2025-10-03",
]

# ============================================================================
# UTILITIES
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Run model evaluation")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["forecast", "nowcast"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "validation", "test"],
        help="Dataset split",
    )

    parser.add_argument(
        "--model", type=str, required=True, choices=["QRNN", "BNN"], help="Model type"
    )

    parser.add_argument(
        "--bidding_zone",
        type=str,
        required=True,
        choices=["NO1", "NO2", "NO3", "NO4"],
        help="Bidding zone",
    )

    return parser.parse_args()


def validate_args(args):
    if args.dataset == "nowcast" and args.split == "test":
        raise ValueError("Nowcast does not support test split")
    if args.dataset == "forecast" and args.split == "train":
        raise ValueError("Forecast does not support train split")


def load_data(dataset, split, bidding_zone, model):
    path_to_model_cfg = PATH_MODEL_CFG[model](bidding_zone[-1])
    model_cfg = load_config(path_to_model_cfg)

    if split == "test":
        scaler_data_nowcast = load_scaler_params(
            bid_zone=f"ELSPOT {bidding_zone}",
            dataset="nowcast",
            dir_path="../checkpoints",
        )
        data = load_evaluation_forecast_data(
            dates=evaluation_dates,
            scaler_params=scaler_data_nowcast["scaler_params"],
            site_scaler_params=scaler_data_nowcast["site_scaler_params"],
            bid_zone=f"ELSPOT {bidding_zone}",
        )

    # Training or validation data
    else:
        data_cfg = load_config(model_cfg.data.path)
        if dataset == "forecast":
            data_cfg.dataset = "forecast"
            data_cfg.ensemble = SimpleNamespace(members=None, variance=False)
        data = load_dataset(data_cfg)

    return data


# Handles all ensemble members
def predict(model, bidding_zone, data, split):
    filename = CHECKPOINTS[model][f"ELSPOT {bidding_zone}"]
    if model == "BNN":
        preds, median, pi = test_BNN_new(
            file_name=filename, eval_data=data, split=DATA_SPLIT[split]
        )

    else:
        preds, median, pi = test_QR_new(
            file_name=filename,
            eval_data=data,
            split=DATA_SPLIT[split],
        )
    return preds, median, pi


# TODO: nur für eine bidding zone, nicht average
def evaluate(model, preds, median, pi, data, split, num_quantiles):

    target = data[f"Y_{DATA_SPLIT[split]}_gen"]

    errors = {}
    errors["rmse"] = round(np.sqrt(np.mean((median - target) ** 2)).item(), 2)

    # Calculate CRPS depending on the model
    if model == "BNN":

        errors["crps"] = round(
            np.mean(ps.crps_ensemble(target, preds, axis=0)).item(), 2
        )
    else:
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        quantiles = torch.linspace(0, 1, num_quantiles)
        errors["crps"] = round(crps_from_quantiles(target, preds, quantiles).item(), 2)

    return errors


def save_result(
    csv_path,
    model,
    split,
    bidding_zone,
    rmse,
    crps,
):
    csv_path = Path(csv_path)

    new_row = pd.DataFrame(
        [
            {
                "model": model,
                "split": split,
                "bidding_zone": bidding_zone,
                "rmse": rmse,
                "crps": crps,
            }
        ]
    )

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Replace if combination already exists
        mask = (
            (df["model"] == model)
            & (df["split"] == split)
            & (df["bidding_zone"] == bidding_zone)
        )

        df = df.loc[~mask]
        df = pd.concat([df, new_row], ignore_index=True)

    else:
        df = new_row

    df = df.sort_values(["model", "split", "bidding_zone"])
    df.to_csv(csv_path, index=False)


# ============================================================================
# MAIN LOOP
# ============================================================================


def main():
    args = parse_args()
    validate_args(args)

    print("Running evaluation with:")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Split        : {args.split}")
    print(f"  Model        : {args.model}")
    print(f"  Bidding Zone : {args.bidding_zone}")

    # Step 1: Load data
    data = load_data(args.dataset, args.split, args.bidding_zone, args.model)

    # Step 2: Load model
    preds, median, pi = predict(args.model, args.bidding_zone, data, args.split)

    # Step 3: Run evaluation
    results = evaluate(
        args.model,
        preds,
        median,
        pi,
        data,
        args.split,
        (
            load_config(
                PATH_MODEL_CFG[args.model](args.bidding_zone[-1])
            ).params.num_quantiles
            if args.model == "QRNN"
            else None
        ),
    )

    # Step 4: Save results
    save_result(
        csv_path="results/evaluation_results.csv",
        model=args.model,
        split=args.split,
        bidding_zone=args.bidding_zone,
        rmse=results["rmse"],
        crps=results["crps"],
    )

    print("Results :")
    print(f"    RMSE: {results["rmse"]}")
    print(f"    CRPS: {results["crps"]}")

    # df = pd.read_csv("results/evaluation_results.csv")
    # wide = df.pivot_table(
    #     index=["model", "split"],
    #     columns="bidding_zone",
    #     values=["rmse", "crps"],
    # )
    # print(wide)


if __name__ == "__main__":
    main()
