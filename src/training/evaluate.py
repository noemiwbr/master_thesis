import torch
import numpy as np
from numpyro.infer import Predictive
from data_processing.preprocess import (
    to_tensor,
)
import arviz as az
import pandas as pd
from jax import random


from data_processing.utils import (
    load_checkpoint,
    load_torch_model,
    QuantileDistribution,
)
from models.model import (
    model_mw_gamma_extended,
)
import torch
import re

import properscoring as ps
from data_processing.utils import QuantileDistribution


def predict(model, data, split="test", member=0):

    X = data[f"X_{split}"]

    # Handle ensemble input, choose specified member
    if X.ndim == 4:
        X = X[:, :, member]

    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    return y_pred


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def rmse_on_quantile(y_true, y_pred_q, quantiles, q=0.5):
    q_idx = torch.argmin(torch.abs(quantiles - q))
    return rmse(y_true, y_pred_q[:, q_idx])


# Uses closest quantile if the requested doesn't exist
def coverage(y_true, y_pred_q, quantiles, lower=0.025, upper=0.975):
    ql = torch.argmin(torch.abs(quantiles - lower))
    qu = torch.argmin(torch.abs(quantiles - upper))

    return torch.mean(
        ((y_true >= y_pred_q[:, ql]) & (y_true <= y_pred_q[:, qu])).float()
    )


def evaluate_model(model, data, quantiles, model_cfg, member=0):
    data = to_tensor(data)

    y_true = data["Y_test_gen"]
    y_true_train = data["Y_train_gen"]

    y_pred = predict(model, data, split="test", member=member)
    y_pred_train = predict(model, data, split="train", member=member)

    metrics = {}
    for s, y_p, y_t in zip(
        ["", "_train"], [y_pred, y_pred_train], [y_true, y_true_train]
    ):
        metrics["rmse_median" + s] = rmse_on_quantile(y_t, y_p, quantiles, q=0.5)
        metrics["coverage_025_975" + s] = coverage(y_t, y_p, quantiles=quantiles)
        metrics["coverage_0_50" + s] = coverage(
            y_t, y_p, quantiles=quantiles, lower=0, upper=0.5
        )
        metrics["crps" + s] = crps_from_quantiles(y_t, y_p, quantiles)

    print("\nTraining statistics:")
    print("   RMSE train:", round(metrics["rmse_median_train"].item(), 3))
    print("   CRPS train:", round(metrics["crps_train"].item(), 3))
    print(
        "   Coverage 2.5-97.5 train:",
        round(metrics["coverage_025_975_train"].item(), 3),
    )
    print("   Coverage 0-50 train:", round(metrics["coverage_0_50_train"].item(), 3))

    return metrics, y_pred, y_pred_train


def evaluate_numpyro(
    model,
    rng_key_predict,
    data,
    model_cfg,
    samples=None,
    member=0,
):

    X_test = data["X_test"]
    X_train = data["X_train"]

    Y_test = data["Y_test_gen"]
    Y_train = data["Y_train_gen"]

    # Handle forecast ensembles, choose specified ensemble member
    if X_test.ndim == 4:
        X_test = X_test[:, :, member]
        X_train = X_train[:, :, member]

    # MCMC
    predictive = Predictive(
        model,
        samples,
        return_sites=["mu", "alpha", "Y"],
    )

    pp_samples = predictive(rng_key_predict, X=X_test, Y=None)
    pp_samples_train = predictive(rng_key_predict, X=X_train, Y=None)

    y_pred = pp_samples["Y"]
    y_pred_train = pp_samples_train["Y"]

    # Metrics: train set
    y_pred_median_train = np.percentile(y_pred_train, 50, axis=0)
    print("\nTraining statistics:")
    print("   RMSE train:", np.sqrt(np.mean((y_pred_median_train - Y_train) ** 2)))
    print("   CRPS train: ", ps.crps_ensemble(Y_train, y_pred_train, axis=0).mean())

    # Metrics: validation set
    y_pred_median = np.percentile(y_pred, 50, axis=0)
    percentiles = np.percentile(y_pred, [2.5, 97.5], axis=0)
    hdis = az.hdi(np.expand_dims(y_pred, axis=0), hdi_prob=0.95).T

    metrics = {}

    metrics["rmse_median"] = np.sqrt(np.mean((y_pred_median - Y_test) ** 2))
    metrics["coverage_025_975"] = ((Y_test >= hdis[0]) & (Y_test <= hdis[1])).mean()
    metrics["coverage_0_50"] = (Y_test <= y_pred_median).mean()
    metrics["crps"] = ps.crps_ensemble(Y_test, y_pred, axis=0).mean()

    # Percentiles shape: [t_test, 2]
    return metrics, y_pred, pp_samples, y_pred_median, percentiles.T, hdis


def test_BNN(file_name, eval_data, member=0, split="eval"):

    eval_data_x = eval_data[f"X_{split}"]
    if len(eval_data_x.shape) == 4:
        eval_data_x = eval_data_x[:, :, member]
        print(f"Choose ensemble member {member}")

    bid_zones = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
    bid_zones = [bz.replace(" ", "_") for bz in bid_zones]

    forecast = pd.DataFrame(
        index=(
            eval_data["index"]
            if split == "eval"
            else eval_data[f"X_{split}_timestamps"]
        ),
        columns=bid_zones,
        data=0,
    )
    forecast_extended = [
        pd.DataFrame(
            index=(
                eval_data["index"]
                if split == "eval"
                else eval_data[f"X_{split}_timestamps"]
            ),
            columns=["median", "lower", "upper"],
            data=0,
        )
        for _ in range(len(bid_zones))
    ]
    model_mapping = {
        "model_mw_gamma_extended": model_mw_gamma_extended,
    }

    model_name = re.match(r"^(.*?)_\d{4}-\d{2}-\d{2}", file_name).group(1)

    for i, bz in enumerate(bid_zones):

        # Load bid zone specific trained model checkpoint
        # TODO make it bidzone specific
        bz_from_checkpoint = re.search(r"(ELSPOT_[A-Z0-9]+)$", file_name).group(1)

        if bz != bz_from_checkpoint:
            continue
        checkpoint = load_checkpoint(
            file_name=file_name
        )  # TODO make it bid zone specific

        # Run inference
        _, rng_key_predict = random.split(checkpoint["rng_key"])

        predictive = Predictive(model_mapping[model_name], checkpoint["samples"])(
            rng_key_predict, eval_data_x
        )

        preds = predictive["Y"]

        median = np.percentile(preds, 50, axis=0)
        hdis = az.hdi(np.expand_dims(preds, axis=0), hdi_prob=0.95)
        hdi_lower, hdi_upper = hdis[:, 0], hdis[:, 1]

        forecast[bz] = median
        forecast_extended[i][["median", "lower", "upper"]] = np.column_stack(
            [median, hdi_lower, hdi_upper]
        )

        # TODO no checkpoints for other bid zones so far

    return forecast, forecast_extended, bz_from_checkpoint, preds


def test_QR(
    file_name,
    eval_data,
    member=0,
    split="eval",
):
    bid_zones = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
    bid_zones = [bz.replace(" ", "_") for bz in bid_zones]

    eval_data_x = eval_data[f"X_{split}"]
    # Handle forecast by choosing specified member
    if len(eval_data[f"X_{split}"].shape) == 4:
        eval_data_x = eval_data[f"X_{split}"][:, :, member]

    forecast = pd.DataFrame(
        index=(
            eval_data["index"]
            if split == "eval"
            else eval_data[f"X_{split}_timestamps"]
        ),
        columns=bid_zones,
        data=0,
    )
    forecast_extended = [
        pd.DataFrame(
            index=(
                eval_data["index"]
                if split == "eval"
                else eval_data[f"X_{split}_timestamps"]
            ),
            columns=["median", "lower", "upper"],
            data=0,
        )
        for _ in range(len(bid_zones))
    ]

    for i, bz in enumerate(bid_zones):
        # extract bid zone
        # TODO only for one bid zone so far
        bz_from_checkpoint = re.search(r"(ELSPOT_[A-Z0-9]+)$", file_name).group(1)
        if bz != bz_from_checkpoint:  # (int(file_name.split("NO")[-1]) - 1):
            continue

        loaded_model, checkpoint_data = load_torch_model(file_name=file_name)
        model_cfg = checkpoint_data["model_cfg"]

        if model_cfg.params.num_quantiles is not None:
            quantiles = np.linspace(0, 1, model_cfg.params.num_quantiles)
        else:
            quantiles = np.linspace(0, 1, 51)

        loaded_model.eval()
        with torch.no_grad():

            y_sum_pred = loaded_model(torch.tensor(eval_data_x))

            Y_pred = y_sum_pred.numpy()

            lower = np.array(
                [
                    QuantileDistribution(quantiles, Y_pred[i]).icdf(0.025)
                    for i in range(eval_data[f"Y_{split}_gen"].shape[0])
                ]
            )
            upper = np.array(
                [
                    QuantileDistribution(quantiles, Y_pred[i]).icdf(0.975)
                    for i in range(eval_data[f"Y_{split}_gen"].shape[0])
                ]
            )

            medians = np.array(
                [
                    QuantileDistribution(quantiles, Y_pred[i]).icdf(0.5)
                    for i in range(eval_data[f"Y_{split}_gen"].shape[0])
                ]
            )

            # TODO: not HDI but percentiles, is naming important of forecast columns?
            forecast_extended[i][["median", "lower", "upper"]] = np.column_stack(
                [medians, lower, upper]
            )

        forecast[bz] = medians

    return forecast, forecast_extended, bz_from_checkpoint, y_sum_pred


def evaluate_numpyro_on_train_set(
    model, samples, rng_key_predict, data, model_cfg, split="train"
):

    X_test = data[f"X_{split}"]
    Y = data[f"Y_{split}_gen"]

    assert (
        X_test.ndim == 3
    ), "Wrong ndim of data, got X_test.ndim={X_test.ndim}, expected 3."

    predictive = Predictive(model, samples)(rng_key_predict, X_test)
    y_pred = predictive["Y"]

    y_pred_median = np.percentile(y_pred, 50, axis=0)
    percentiles = np.percentile(y_pred, [2.5, 97.5], axis=0)
    hdis = az.hdi(np.expand_dims(y_pred, axis=0), hdi_prob=0.95).T

    metrics = {}

    metrics["rmse_median"] = np.sqrt(np.mean((y_pred_median - Y) ** 2))
    metrics["crps"] = ps.crps_ensemble(Y, y_pred, axis=0).mean()
    metrics["coverage_025_975"] = ((Y >= hdis[:, 0]) & (Y <= hdis[:, 1])).mean()
    metrics["coverage_0_50"] = (Y <= y_pred_median).mean()

    # Percentiles shape: [t_test, 2]
    return metrics, y_pred, predictive, y_pred_median, percentiles.T, hdis


def crps_from_quantiles(Y, Q, taus):
    """
    Y:    [N,]
    Q:    [N,Q]
    taus: [Q,]
    """
    Y = Y.unsqueeze(1)
    diff = Y - Q

    # Pinball loss
    loss = torch.maximum(taus * diff, (taus - 1) * diff)

    # Integration over quantiles
    crps = 2 * torch.trapezoid(loss, taus, dim=1)  # [N,]

    return crps.mean()
