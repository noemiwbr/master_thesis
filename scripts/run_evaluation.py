import sys

sys.path.append("../src")

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
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
from training.evaluate import (
    test_BNN_new,
    test_QR_new,
    crps_from_quantiles,
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

# For shifted baselines, load those dates for the shifted data
add_evaluation_dates_bl = [
    "2025-09-17",
    "2025-09-18",
    "2025-09-19",
    "2025-09-20",
    "2025-09-21",
    "2025-09-27",
    "2025-09-28",
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
        "--model",
        type=str,
        required=True,
        choices=["QRNN", "BNN", "shifted", "median"],
        help="Model type",
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

    if split == "test":
        scaler_data_nowcast = load_scaler_params(
            bid_zone=f"ELSPOT {bidding_zone}",
            dataset="nowcast",
            dir_path="../checkpoints",
        )
        if model == "shifted":
            eval_dates = evaluation_dates + add_evaluation_dates_bl
            eval_dates.sort()

        else:
            eval_dates = evaluation_dates

        data = load_evaluation_forecast_data(
            dates=eval_dates,
            scaler_params=scaler_data_nowcast["scaler_params"],
            site_scaler_params=scaler_data_nowcast["site_scaler_params"],
            bid_zone=f"ELSPOT {bidding_zone}",
        )

    # Training or validation data
    else:
        data_cfg = load_config(
            f"../configs/data/nowcast0{bidding_zone[-1]}.yaml"
        )  # load_config(model_cfg.data.path)
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


def evaluate(model, preds, median, data, split, num_quantiles):

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


def evaluate_baselines(model, data, split, bidding_zone):
    errors = {}

    if model == "shifted":
        if split == "test":
            evaluation_dates_d = pd.to_datetime(evaluation_dates)
            windpower_bz = pd.DataFrame(
                data["Y_eval_gen"],
                index=data["index"],
                columns=[f"ELSPOT {bidding_zone}"],
            )
            s = windpower_bz.sort_index()
            s_hourly = s.asfreq("h")
            s["lag_3D"] = s_hourly.shift(72)
            s = s.dropna()
            s_filtered = s[np.isin(s.index.date, evaluation_dates_d.date)]
            s_filtered_na = s_filtered.dropna()

        else:
            windpower = load_windpower("../data/wind_power_per_bidzone.parquet")
            windpower_bz = pd.DataFrame(windpower[f"ELSPOT {bidding_zone}"])
            s = windpower_bz.sort_index()
            s_hourly = s.asfreq("h")
            timestamps = pd.Index(data[f"X_{DATA_SPLIT[split]}_timestamps"])
            s_hourly.loc[~s_hourly.index.isin(timestamps)] = np.nan
            s_filtered = s_hourly.loc[
                data[f"X_{DATA_SPLIT[split]}_timestamps"][0] : data[
                    f"X_{DATA_SPLIT[split]}_timestamps"
                ][-1]
            ].copy()
            s_filtered["lag_3D"] = s_filtered.iloc[:, 0].shift(72)

            s_filtered_na = s_filtered.dropna()
        errors["rmse"] = round(
            np.sqrt(
                np.mean(
                    (s_filtered_na[f"ELSPOT {bidding_zone}"] - s_filtered_na["lag_3D"])
                    ** 2
                )
            ),
            2,
        )

        errors["crps"] = round(
            np.mean(
                ps.crps_ensemble(
                    s_filtered_na[f"ELSPOT {bidding_zone}"], s_filtered_na["lag_3D"]
                )
            ).item(),
            2,
        )

    else:  # model == "median"
        key = f"Y_{DATA_SPLIT[split]}_gen"

        # Calculate median on training set
        data_cfg_bl = load_config(f"../configs/data/nowcast0{bidding_zone[-1]}.yaml")
        data_bl = load_dataset(data_cfg_bl)
        median_pred = np.median(data_bl["Y_train_gen"])

        errors["rmse"] = round(
            np.sqrt(
                np.mean(
                    (data[key] - np.broadcast_to(median_pred, data[key].shape)) ** 2
                )
            ).item(),
            2,
        )
        errors["crps"] = round(
            np.mean(
                ps.crps_ensemble(
                    data[key],
                    np.broadcast_to(median_pred, data[key].shape),
                )
            ).item(),
            2,
        )
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

    if args.model in ["QRNN", "BNN"]:
        # Step 2: Load model
        preds, median, pi = predict(args.model, args.bidding_zone, data, args.split)

        # Step 3: Run evaluation
        results = evaluate(
            args.model,
            preds,
            median,
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
    elif args.model in ["median", "shifted"]:
        # Step 2 + 3
        results = evaluate_baselines(args.model, data, args.split, args.bidding_zone)

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
