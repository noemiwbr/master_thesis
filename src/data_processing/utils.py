import os
import pickle
from datetime import datetime
from glob import glob
import time
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import pandas as pd
from models.model import (
    WindPowerQuantile,
    WindPowerModel_simple,
    WindPowerQuantileEnsembles,
    WindPowerQuantile_weighted,
)
from data_processing.load import (
    load_weather_forecast_parquet,
    rename_met_forecast_columns,
)
import sys
import glob
from datetime import datetime


def save_checkpoint(
    samples,
    model_cfg,
    rng_key=None,
    rng_key_predict=None,
    dir_path="../checkpoints",
    dataset="forecast",
):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bid_zone_clean = model_cfg.data.bid_zone.replace(" ", "_")
    filename = f"{model_cfg.name}_{timestamp}_{dataset}_{bid_zone_clean}.pkl"
    path = os.path.join(dir_path, filename)

    # Bundle all relevant information into one dictionary
    state = {
        "samples": samples,
        "rng_key": rng_key,
        "rng_key_predict": rng_key_predict,
        "model_cfg": model_cfg,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved checkpoint for model '{model_cfg.name}': {path}")
    return path


# Save scaler parameters for a specific bid zone and split time
def save_scaler_params(
    scaler_params,
    site_scaler_params,
    bid_zone="ELSPOT_NO1",
    split_time=None,
    dataset="nowcast",
    dir_path="../checkpoints",
):

    os.makedirs(dir_path, exist_ok=True)

    # Clean bid zone name for filename
    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"scaler_params_{bid_zone_clean}_{dataset}.pkl"
    path = os.path.join(dir_path, filename)

    state = {
        "bid_zone": bid_zone,
        "split_time": str(split_time) if split_time is not None else None,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scaler_params": scaler_params,
        "site_scaler_params": site_scaler_params,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved scaler parameters for {bid_zone}: {path}")
    return path


def load_checkpoint(model_name=None, file_name=None, dir_path="../checkpoints"):

    if file_name is not None:
        path_to_ckp = os.path.join(dir_path, f"{file_name}.pkl")
        print(f"Loading specified checkpoint: {path_to_ckp}")

    else:
        pattern = os.path.join(dir_path, f"{model_name}_*.pkl")
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No checkpoint files found for model '{model_name}' in '{dir_path}'"
            )

        # Sort by modification time and pick the latest
        files.sort(key=os.path.getmtime)
        path_to_ckp = files[-1]

        print(f"Loading latest checkpoint: {path_to_ckp}")

    with open(path_to_ckp, "rb") as f:
        state = pickle.load(f)

    return state


def save_torch_checkpoint(
    model,
    model_cfg,
    optimizer,
    num_features,
    losses,
    dataset="forecast",
    dir_path="../checkpoints",
    num_parks=None,
):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    bid_zone_clean = model_cfg.data.bid_zone.replace(" ", "_")
    filename = f"{model_cfg.name}_{timestamp}_{dataset}_{model_cfg.params.target}_{bid_zone_clean}.pt"
    path = os.path.join(dir_path, filename)

    # Save model weights, optimizer state, and model configuration
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_cfg": model_cfg,
        "num_features": num_features,
        "num_parks": num_parks,
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {filename}")

    return path


def load_torch_model(
    file_name=None,
    model_name=None,
    dir_path="../checkpoints",
    quantiles=None,
):

    if file_name is not None:
        path = os.path.join(dir_path, f"{file_name}.pt")
    else:
        pattern = os.path.join(dir_path, f"{model_name}_*.pt")
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No checkpoint files found for model '{model_name}' in '{dir_path}'"
            )

        files.sort(key=os.path.getmtime)
        path = files[-1]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model_class_name = checkpoint["model_cfg"].name
    num_features = checkpoint["num_features"]
    num_parks = checkpoint["num_parks"]
    target = checkpoint["model_cfg"].params.target

    # Quantiles
    if quantiles is not None:
        quantiles = quantiles
    elif checkpoint["model_cfg"].params.quantiles is None:
        quantiles = torch.linspace(0, 1, 51)
    else:
        quantiles = torch.tensor(checkpoint["model_cfg"].params.quantiles)

    if model_class_name == "WindPowerQuantile":
        model = WindPowerQuantile(num_features, quantiles, target)
    elif model_class_name == "WindPowerModel_simple":
        model = WindPowerModel_simple(num_features)
    elif model_class_name == "WindPowerQuantile_weighted":
        model = WindPowerQuantile_weighted(
            num_parks=num_parks, num_features=num_features, quantiles=quantiles
        )
    elif model_class_name == "WindPowerQuantileEnsembles":
        model = WindPowerQuantileEnsembles(num_features, quantiles, target)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: {os.path.basename(path)}")
    if "bid_zone" in checkpoint:
        print(f"  Bid zone: {checkpoint["model_cfg"].data.bid_zone}")
    print(f"  Model class: {model_class_name}")

    return model, checkpoint


# Inference helper for basic BNN
def run_inference_bnn(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        chain_method="parallel",
        progress_bar=sys.stdout.isatty(),
    )
    mcmc.run(
        rng_key,
        X=X,
        Y=Y,
        extra_fields=("accept_prob", "num_steps", "diverging"),
    )
    mcmc.print_summary()
    extras = mcmc.get_extra_fields()
    print("Avg. accept prob:", extras["accept_prob"].mean())
    print("Total divergences:", extras["diverging"].sum())
    print("Avg. NUTS steps:", extras["num_steps"].mean())
    print("\nMCMC elapsed time:", time.time() - start)
    samples = mcmc.get_samples()
    return samples


# Posterior predictive for basic BNN
def predict_bnn(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(model).get_trace(X=X, Y=None)
    return trace["Y"]["value"]


class QuantileDistribution:
    """
    A utility class that converts model-predicted quantiles q(tau)
    into a full predictive distribution with:
        - inverse CDF
        - CDF
        - PDF (via dq/dtau)
        - sampling
        - highest-density intervals (HDIs)
        - KDE-based PDF for smoothing
        - plotting utilities
    """

    def __init__(self, quantile_levels, quantile_values, eps=1e-6):
        self.taus = np.asarray(quantile_levels)
        self.qvals = np.asarray(quantile_values)
        self.eps = eps

        # Sort to ensure correct order
        idx = np.argsort(self.taus)
        self.taus = self.taus[idx]
        self.qvals = self.qvals[idx]

        assert len(self.taus) == len(self.qvals), "taus and qvals must have same length"

        # Enforce monotonicity (usually already guaranteed by model)
        self.qvals = np.maximum.accumulate(self.qvals)

    def icdf(self, tau):
        """Return q(tau) by interpolation."""
        return np.interp(tau, self.taus, self.qvals)

    def cdf(self, y):
        """Return F(y) via interpolation τ = F(y)."""
        return np.interp(y, self.qvals, self.taus, left=0.0, right=1.0)

    def pdf(self):
        """
        PDF via dq/dtau, non-smooth or unstable for flat CDF
        Returns:
            x: q-values (same as quantile_values)
            pdf_vals: approximate PDF evaluated at q-values
        """
        q = self.qvals
        t = self.taus
        K = len(q)

        # Derivative dq/dτ
        slope = (q[1:] - q[:-1]) / (t[1:] - t[:-1])  # Length K-1

        # Pad to length K
        slope = np.concatenate([[slope[0]], slope, [slope[-1]]])
        slope = slope[:K]

        pdf_vals = 1.0 / (slope + self.eps)
        return q, pdf_vals

    def sample(self, n_samples=2000):
        """Inverse-transform sampling: Y = q(U), U ~ Uniform(0,1)"""
        u = np.random.rand(n_samples)
        return np.interp(u, self.taus, self.qvals)

    def hdi(self, cred_mass=0.95, n_samples=20000):
        """
        Compute HDI (shortest interval containing 'cred_mass' of the probability)
        Implemented via sampling
        Returns:
            (hdi_min, hdi_max)
        """
        samples = self.sample(n_samples)
        samples = np.sort(samples)

        m = int(np.floor(cred_mass * n_samples))

        if m < 1:
            raise ValueError("cred_mass too small or not enough samples")

        # Candidate intervals, each candidate contains m samples (i.e. 95% mass)
        widths = samples[m:] - samples[: n_samples - m]

        # Find shortest interval -> HDI
        idx_min = np.argmin(widths)

        return samples[idx_min], samples[idx_min + m]

    def kde_pdf(self, n_samples=5000, grid_size=300):
        """
        Smooth PDF using KDE
        Returns:
            x_grid, kde_pdf_values
        """
        samples = self.sample(n_samples)
        kde = gaussian_kde(samples)

        x_min, x_max = samples.min(), samples.max()
        x_grid = np.linspace(x_min, x_max, grid_size)
        pdf_vals = kde(x_grid)
        return x_grid, pdf_vals

    def plot_cdf(self, ax=None, **kwargs):
        x, y = self.qvals, self.taus
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, **kwargs)
        ax.set_xlabel("Value")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
        ax.set_title("Estimated CDF")
        return ax

    def plot_pdf(self, ax=None, **kwargs):
        x, pdf_vals = self.pdf()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, pdf_vals, **kwargs)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.set_title("Approximate PDF (dq/dτ)")
        return ax

    def plot_kde_pdf(self, n_samples=5000, grid_size=300, bins=40):
        samples = self.sample(n_samples)
        x_grid, pdf_vals = self.kde_pdf(n_samples, grid_size)

        plt.figure(figsize=(6, 4))
        plt.hist(samples, bins=bins, density=True, alpha=0.4, label="Samples")
        plt.plot(x_grid, pdf_vals, lw=2, label="KDE PDF")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title("Smooth PDF via KDE")
        plt.show()


def approximate_qr_intervals(
    Y_pred,
    taus,
    cred_mass=0.95,
    n_samples=5000,
):
    T = Y_pred.shape[0]

    hdis = np.empty((T, 2))
    medians = np.empty(T)
    q_lower = np.empty(T)
    q_upper = np.empty(T)

    for i in range(T):
        qd = QuantileDistribution(taus, Y_pred[i])
        hdis[i] = qd.hdi(cred_mass=cred_mass, n_samples=n_samples)
        medians[i] = qd.icdf(0.5)
        q_lower[i] = qd.icdf((1 - cred_mass) / 2)
        q_upper[i] = qd.icdf(1 - (1 - cred_mass) / 2)

    return hdis, medians, q_lower, q_upper


# Compute ensemble variance per lead time for multiple features.
def compute_ensemble_variance_by_lead_time():
    forecast = load_weather_forecast_parquet("../data/met_forecast.parquet")
    forecast = rename_met_forecast_columns(forecast)
    training_mask = forecast["time"] < pd.Timestamp("2025-01-01 00:00:00", tz="CET")
    forecast = forecast.loc[training_mask]

    feature_cols = [
        "air_temperature_2m",
        "air_pressure_at_sea_level",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]
    feature_cols.sort()
    # Map feature → member columns
    feature_member_cols = {
        feat: [f"{feat}_member_{i}" for i in range(15)] for feat in feature_cols
    }

    var_by_feature = {}

    for feat, cols in feature_member_cols.items():
        # Variance across members per windpark & lead time
        df_var = (
            forecast.groupby(["windpark", "lead_time_hours"])[cols]
            .var()
            .mean(axis=1)
            .reset_index(name="var")
        )

        # Average across windparks
        avg_var = df_var.groupby("lead_time_hours")["var"].mean()
        var_by_feature[feat] = avg_var

    # Combine features into DataFrame
    var_overall_features = pd.DataFrame(var_by_feature)

    # Overall variance across all features and members
    all_member_cols = [f"{feat}_member_{i}" for feat in feature_cols for i in range(15)]

    overall_variance = (
        forecast.groupby("lead_time_hours")[all_member_cols].var().mean(axis=1)
    )

    return var_by_feature, var_overall_features, overall_variance


def extract_test_dates(num=30, data_path="../data/evaluation/", cut_off="2025-09-22"):
    """
    Evaluation dates (#num) for representative test set, cut_off determines until which date the data can serve as test set
    """
    extract_date_weather = lambda f: datetime.strptime(
        f.split("/")[-1].split("_")[2][:8], "%Y%m%d"
    )
    extract_date_power = lambda f: datetime.strptime(
        f.split("/")[-1].split("_")[-1].replace(".parquet", ""), "%Y%m%d"
    )

    files_weather = glob.glob(data_path + "/met_forecast_daily/*.csv")
    files_power = glob.glob(data_path + "/wind_power/*.parquet")

    dates_weather = [extract_date_weather(f) for f in files_weather]
    dates_weather = sorted(dates_weather)
    dates_power = [extract_date_power(f) for f in files_power]
    dates_power = sorted(dates_power)

    date_strings_weather = [d.strftime("%Y-%m-%d") for d in dates_weather]
    date_strings_power = [d.strftime("%Y-%m-%d") for d in dates_power]

    dates = [date for date in date_strings_weather if date in date_strings_power]

    actual_test_set_idx = dates.index(cut_off)
    valid_dates = dates[:actual_test_set_idx]
    return valid_dates[-num:]
