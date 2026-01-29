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
from models.model import model_mw_gamma, model_cf_logitnormal, model_cf_beta
import torch
import re


def predict(model, data, model_cfg, split="test"):

    X = data[f"X_{split}"]

    # Model-specific input handling
    if model_cfg.name == "WindPowerQuantile" and X.ndim == 4:
        X = X[:, :, 0]

    if model_cfg.name == "WindPowerQuantileEnsembles" and X.ndim != 4:
        raise ValueError(
            "WindPowerQuantileEnsembles requires ensemble input with shape "
            "[T, S, E, F], but got X.ndim = "
            f"{X.ndim} and shape {tuple(X.shape)}"
        )

    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    return y_pred


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def rmse_on_quantile(y_true, y_pred_q, quantiles, q=0.5):
    q_idx = torch.argmin(torch.abs(quantiles - q))
    return rmse(y_true, y_pred_q[:, q_idx])


def coverage(y_true, y_pred_q, quantiles, lower=0.025, upper=0.975):
    ql = torch.argmin(torch.abs(quantiles - lower))
    qu = torch.argmin(torch.abs(quantiles - upper))

    return torch.mean(
        ((y_true >= y_pred_q[:, ql]) & (y_true <= y_pred_q[:, qu])).float()
    )


def evaluate_model(model, data, quantiles, model_cfg, data_cfg):
    data = to_tensor(data, data_cfg)

    y_true = (
        data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]
    )

    y_pred = predict(model, data, model_cfg, split="test")

    metrics = {}
    if model_cfg.name == "WindPowerModel_simple":  # no quantiles
        metrics["rmse_median"] = rmse(y_true, y_pred)
        metrics["coverage_025_975"] = 0
        metrics["coverage_0_50"] = 0
    else:
        metrics["rmse_median"] = rmse_on_quantile(y_true, y_pred, quantiles, q=0.5)
        metrics["coverage_025_975"] = coverage(y_true, y_pred, quantiles=quantiles)
        metrics["coverage_0_50"] = coverage(
            y_true, y_pred, quantiles=quantiles, lower=0, upper=0.5
        )

    return metrics, y_pred


def evaluate_numpyro(model, samples, rng_key_predict, data, model_cfg):

    X_test = data["X_test"]
    X_train = data["X_train"]
    Y = data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]

    # Handle forecast vs nowcast
    if model_cfg.name in ["model_mw_gamma", "model_cf_beta"] and X_test.ndim == 4:
        X_test = X_test[:, :, 0]  # first ensemble member
        X_train = X_train[:, :, 0]

    predictive = Predictive(model, samples)(rng_key_predict, X_test)

    # Print the RMSE on the training data - sanity check that the model learns something
    Y_train = (
        data["Y_train_gen"] if model_cfg.params.target == "gen" else data["Y_train_cf"]
    )
    predictive_train = Predictive(model, samples)(rng_key_predict, X_train)
    y_pred_train = (
        predictive_train["Y"]
        if model_cfg.params.target == "gen"
        else predictive_train["Y"] * data["Y_train_aval"]
    )
    y_pred_median_train = np.percentile(y_pred_train, 50, axis=0)
    print("RMSE train:", np.sqrt(np.mean((y_pred_median_train - Y_train) ** 2)))

    # Handle CF prediction, transform it back to generation prediction
    y_pred = (
        predictive["Y"]
        if model_cfg.params.target == "gen"
        else predictive["Y"] * data["Y_test_aval"]
    )
    y_pred_median = np.percentile(y_pred, 50, axis=0)
    percentiles = np.percentile(y_pred, [2.5, 97.5], axis=0)
    hdis = az.hdi(np.expand_dims(y_pred, axis=0), hdi_prob=0.95).T

    metrics = {}

    metrics["rmse_median"] = np.sqrt(np.mean((y_pred_median - Y) ** 2))
    metrics["coverage_025_975"] = ((Y >= percentiles[0]) & (Y <= percentiles[1])).mean()
    metrics["coverage_0_50"] = (Y <= y_pred_median).mean()

    # Percentiles shape: [t_test, 2]
    return metrics, y_pred, predictive, y_pred_median, percentiles.T, hdis


def test_BNN(file_name, eval_data):

    if len(eval_data["X_eval"].shape) == 4:
        eval_data["X_eval"] = eval_data["X_eval"][:, :, 0]

    bid_zones = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
    bid_zones = [bz.replace(" ", "_") for bz in bid_zones]

    forecast = pd.DataFrame(
        index=eval_data["index"],
        columns=bid_zones,
        data=0,
    )
    forecast_extended = [
        pd.DataFrame(
            index=eval_data["index"],
            columns=["median", "hdi_lower", "hdi_upper"],
            data=0,
        )
        for _ in range(len(bid_zones))
    ]
    model_mapping = {
        "model_cf_beta": model_cf_beta,
        "model_mw_gamma": model_mw_gamma,
        "model_cf_logitnormal": model_cf_logitnormal,
    }

    for i, bz in enumerate(bid_zones):

        # Load bid zone specific trained model checkpoint
        # TODO make it bidzone specific
        model_name = re.match(r"^(.*?)_\d{4}-\d{2}-\d{2}", file_name).group(1)

        checkpoint = load_checkpoint(
            file_name=file_name
        )  # TODO make it bid zone specific

        # Run inference
        _, rng_key_predict = random.split(checkpoint["rng_key"])

        predictive = Predictive(model_mapping[model_name], checkpoint["samples"])(
            rng_key_predict, eval_data["X_eval"]
        )

        preds = predictive["Y"]
        if "cf" in checkpoint["model_cfg"].name:
            preds *= eval_data["Y_eval_aval"]

        median = np.percentile(preds, 50, axis=0)
        hdis = az.hdi(np.expand_dims(preds, axis=0), hdi_prob=0.95)
        hdi_lower, hdi_upper = hdis[:, 0], hdis[:, 1]

        forecast[bz] = median
        forecast_extended[i][["median", "hdi_lower", "hdi_upper"]] = np.column_stack(
            [median, hdi_lower, hdi_upper]
        )

        break  # TODO no checkpoints for other bid zones so far

    return forecast, forecast_extended


def test_QR(file_name, eval_data, ensembles=False, from_now_to_forecast=False):
    bid_zones = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
    bid_zones = [bz.replace(" ", "_") for bz in bid_zones]

    assert (
        from_now_to_forecast and len(eval_data["X_eval"].shape) == 4
    ), "forecast has to have 4 dimensions"

    if from_now_to_forecast:
        feature_cols = eval_data["feature_cols"]
        idx = [i for i, c in enumerate(feature_cols) if c != "avg_ensemble_var"]
        X_eval = eval_data["X_eval"][:, :, 0, idx]

    if len(eval_data["X_eval"].shape) == 4 and not ensembles:
        X_eval = eval_data["X_eval"][:, :, 0]
    else:
        X_eval = eval_data["X_eval"]

    forecast = pd.DataFrame(
        index=eval_data["index"],
        columns=bid_zones,
        data=0,
    )
    forecast_extended = [
        pd.DataFrame(
            index=eval_data["index"],
            columns=["median", "hdi_lower", "hdi_upper"],
            data=0,
        )
        for _ in range(len(bid_zones))
    ]

    for i, bz in enumerate(bid_zones):
        # extract bid zone
        # TODO only for one bid zone so far
        if i != (int(file_name.split("NO")[-1]) - 1):
            continue

        loaded_model, checkpoint_data = load_torch_model(file_name=file_name)

        loaded_model.eval()
        with torch.no_grad():

            if "Quantile" in checkpoint_data["model_cfg"].name:
                y_sum_pred = loaded_model(torch.tensor(X_eval))

                if "cf" in file_name:
                    y_sum_pred *= eval_data["Y_eval_aval"][:, None]
                Y_pred = y_sum_pred.numpy()

                taus = np.linspace(0, 1, 51)
                hdis = np.array(
                    [
                        QuantileDistribution(taus, Y_pred[i]).hdi(
                            cred_mass=0.95, n_samples=5000
                        )
                        for i in range(eval_data["Y_eval_gen"].shape[0])
                    ]
                )

                medians = np.array(
                    [
                        QuantileDistribution(taus, Y_pred[i]).icdf(0.5)
                        for i in range(eval_data["Y_eval_gen"].shape[0])
                    ]
                )

                forecast_extended[i][["median", "hdi_lower", "hdi_upper"]] = (
                    np.column_stack([medians, hdis[:, 0], hdis[:, 1]])
                )

            else:  # TODO catch more models here
                y_sum_pred = loaded_model(
                    torch.tensor(X_eval)
                ).numpy()  # TODO * eval_data["Y_eval_aval"]  # CF
                medians = y_sum_pred  # np.percentile(y_sum_pred, 50, axis=0)
                forecast_extended[i]["median"] = medians

        forecast[bz] = medians

    return forecast, forecast_extended
