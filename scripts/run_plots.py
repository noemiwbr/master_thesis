import sys

sys.path.append("../src")

from types import SimpleNamespace
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import properscoring as ps
import torch

from configs.load_config import load_config

from data_processing.load import (
    load_dataset,
    load_evaluation_forecast_data,
    load_scaler_params,
    load_windpower,
)

from data_processing.utils import load_checkpoint, load_torch_model, save_scaler_params

from models.build import build_model

from training.evaluate import (
    crps_from_quantiles,
    test_BNN_all,
    test_QR_all,
)

from training.plot import (
    agg_weights_plot,
    calibration_diagnostics,
    plot_bnn_gamma_combined,
    plot_qr_hdi_prediction_simple,
    plot_qr_prediction_ensemble,
    window_plot,
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
    "BNN": lambda i: f"../configs/model/bnn_gamma_0{i}.yaml",
    "QRNN": lambda i: f"../configs/model/qrnn_0{i}.yaml",
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

        if data_cfg.dataset == "nowcast" and split == "train":
            save_scaler_params(
                scaler_params=data["scaler_params"],
                site_scaler_params=data["site_scaler_params"],
                bid_zone=data_cfg.bid_zone,
                split_time=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                dataset=data_cfg.dataset,
                dir_path="../checkpoints",
            )

    return data


# Handles all ensemble members
def predict(model, bidding_zone, data, split):
    filename = CHECKPOINTS[model][f"ELSPOT {bidding_zone}"]
    if model == "BNN":
        preds_list, preds, median, pi = test_BNN_all(
            file_name=filename, eval_data=data, split=DATA_SPLIT[split]
        )

    else:
        preds_list, preds, median, pi = test_QR_all(
            file_name=filename,
            eval_data=data,
            split=DATA_SPLIT[split],
        )
    return preds_list, preds, median, pi


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


def plot_nn(dataset, model, split, bidding_zone, data, median, pi, preds_list):
    if dataset == "forecast":
        if model == "QRNN":
            # Plot predictions
            plot_qr_prediction_ensemble(
                Y_true=data[f"Y_{DATA_SPLIT[split]}_gen"],
                Y_pred=preds_list,
                plot_interval=[0, 1080],
                title=f"{model}: Aggregated ensemble member prediction ({split} set, {bidding_zone})",
                timestamps=(
                    data["index"]
                    if split == "test"
                    else data[f"X_{DATA_SPLIT[split]}_timestamps"]
                ),
            )
            plt.savefig(
                f"../plots/ensemble_{dataset}_{model}_{bidding_zone}_{DATA_SPLIT[split]}_font.pdf"
            )
            # plt.show()

        if model == "BNN":
            filename = CHECKPOINTS[model][f"ELSPOT {bidding_zone}"]
            checkpoint = load_checkpoint(file_name=filename)
            BNN = build_model(checkpoint["model_cfg"], data)

            # Plot predictions
            plot_bnn_gamma_combined(
                checkpoint,
                BNN,
                data,
                load_config(checkpoint["model_cfg"].data.path),
                [0, 1080],
                split=DATA_SPLIT[split],
            )
            plt.savefig(
                f"../plots/ensemble_{dataset}_{model}_{bidding_zone}_{DATA_SPLIT[split]}_font.pdf"
            )

    if dataset == "nowcast":
        if model == "QRNN":
            plot_qr_hdi_prediction_simple(
                Y_true=data[f"Y_{DATA_SPLIT[split]}_gen"],
                conf_interval=pi,
                medians=median,
                plot_interval=[0, 1080],
                title=f"{model}: Nowcast prediction ({split} set, {bidding_zone})",
                timestamps=data[f"X_{DATA_SPLIT[split]}_timestamps"],
            )
            plt.savefig(
                f"../plots/{dataset}_{model}_{bidding_zone}_{DATA_SPLIT[split]}.pdf"
            )
            # plt.show()

        if model == "BNN":

            plot_qr_hdi_prediction_simple(
                Y_true=data[f"Y_{DATA_SPLIT[split]}_gen"],
                conf_interval=pi.T,  # hdi_list[0].T,
                medians=median,  # median_list[0],
                plot_interval=[0, 1080],
                title=f"{model}: Nowcast prediction ({split} set, {bidding_zone})",
                timestamps=data[f"X_{DATA_SPLIT[split]}_timestamps"],
            )
            plt.savefig(
                f"../plots/{dataset}_{model}_{bidding_zone}_{DATA_SPLIT[split]}.pdf"
            )
            # plt.show()


def plot_baselines(model, dataset, data, split, bidding_zone, n_plot=1080):

    if model == "shifted" and split == "validation" and dataset == "forecast":
        # Calculate median on training set
        data_cfg_bl = load_config(f"../configs/data/nowcast0{bidding_zone[-1]}.yaml")
        data_bl = load_dataset(data_cfg_bl)
        baseline_median = np.median(data_bl["Y_train_gen"])

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

        plt.figure(figsize=(15, 4))
        plt.plot(
            s_filtered.index[:n_plot],
            s_filtered[f"ELSPOT {bidding_zone}"][:n_plot],
            ".",
            color="black",
            label="Observed data",
            markersize=4,
        )

        plt.plot(
            s_filtered.index[:n_plot],
            np.broadcast_to(baseline_median, (n_plot)),
            color="red",
            label="Baseline 1: Median",
            lw=2,
        )

        plt.plot(
            s_filtered.index[:n_plot],
            s_filtered["lag_3D"][:n_plot],
            color="blue",
            label="Baseline 2: Shifted Wind Power",
            lw=2,
        )
        plt.ylabel("Total power (MW)")
        plt.xlabel("Time")
        plt.legend()
        plt.title(f"Baseline predictions (validation set, ELSPOT {bidding_zone})")
        plt.tight_layout()
        plt.savefig(f"../plots/baselines_ELSPOT_{bidding_zone}.pdf")

    return


def plot_nn_val(model, data, bidding_zone, preds):
    filename = CHECKPOINTS[model][f"ELSPOT {bidding_zone}"]
    model_cfg = load_config(PATH_MODEL_CFG[model](bidding_zone[-1]))

    if model == "QRNN":

        quantiles = torch.linspace(
            0,
            1,
            model_cfg.params.num_quantiles,
        )
        nn, checkpoint = load_torch_model(file_name=filename, quantiles=quantiles)
        learned_weights = torch.softmax(
            checkpoint["model_state_dict"]["agg_logits"], dim=0
        )

    if model == "BNN":
        checkpoint = load_checkpoint(file_name=filename)
        nn = build_model(checkpoint["model_cfg"], data)
        learned_weights = checkpoint["samples"]["agg_logits"].mean(0)

    # Plot aggregation weights
    # agg_weights_plot(learned_weights, data["sites_meta"], model, bidding_zone)
    # plt.savefig(f"../plots/agg_weights_{model}_{bidding_zone}.pdf")

    # # Plot RMSE/CRPS on different plotting windows
    # window_plot(data, model_cfg, checkpoint, nn, model)
    # plt.savefig(f"../plots/metrics_windows_{model}_{bidding_zone}.pdf")

    # Plot calibration diagnostics
    calibration_diagnostics(data, preds, torch.linspace(0, 1, 41), model, bidding_zone)
    plt.savefig(f"../plots/calibration_overview_{model}_{bidding_zone}_font.pdf")


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
        preds_list, preds, median, pi = predict(
            args.model, args.bidding_zone, data, args.split
        )

        plot_nn(
            args.dataset,
            args.model,
            args.split,
            args.bidding_zone,
            data,
            median,
            pi,
            preds_list,
        )

        if args.split == "validation" and args.dataset == "forecast":
            plot_nn_val(args.model, data, args.bidding_zone, preds)

    elif args.model in ["median", "shifted"]:
        # Step 2 + 3
        plot_baselines(
            args.model, args.dataset, data, args.split, args.bidding_zone, n_plot=1080
        )


if __name__ == "__main__":
    main()
