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
from models.model import WindPowerQuantile, WindPowerModel_simple


def save_checkpoint(
    samples,
    model_name="bnn",
    args=None,
    rng_key=None,
    rng_key_predict=None,
    dir_path="../checkpoints",
    target="gen",
    bid_zone="ELSPOT_NO1",
):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_{timestamp}_{target}_{bid_zone}.pkl"
    path = os.path.join(dir_path, filename)

    # Bundle all relevant information into one dictionary
    state = {
        "model_name": model_name,
        "timestamp": timestamp,
        "samples": samples,
        "args": args,
        "rng_key": rng_key,
        "rng_key_predict": rng_key_predict,
        "bid_zone": bid_zone,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved checkpoint for model '{model_name}': {path}")
    return path


# Save scaler parameters for a specific bid zone and split time
def save_scaler_params(
    scaler_params,
    site_scaler_params,
    bid_zone="ELSPOT_NO1",
    split_time=None,
    dir_path="../checkpoints",
):

    os.makedirs(dir_path, exist_ok=True)

    # Clean bid zone name for filename
    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"scaler_params_{bid_zone_clean}.pkl"
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


# Load scaler parameters for a specific bid zone
def load_scaler_params(bid_zone="ELSPOT_NO1", dir_path="../checkpoints"):

    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"scaler_params_{bid_zone_clean}.pkl"
    path = os.path.join(dir_path, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No scaler parameters found for {bid_zone} at {path}\n"
            f"Run load_and_preprocess_data() and save_scaler_params() first."
        )

    with open(path, "rb") as f:
        state = pickle.load(f)

    print(f"Loaded scaler parameters for {state['bid_zone']}")
    print(f"  Saved at: {state['saved_at']}")
    if state["split_time"]:
        print(f"  Split time: {state['split_time']}")

    return {
        "scaler_params": state["scaler_params"],
        "site_scaler_params": state["site_scaler_params"],
    }


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
    optimizer=None,
    model_name="WindPowerModel_simple",
    bid_zone="ELSPOT_NO1",
    quantiles=None,
    num_features=None,
    dir_path="../checkpoints",
):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"{model_name}_{timestamp}_{bid_zone_clean}.pt"
    path = os.path.join(dir_path, filename)

    # Save model weights, optimizer state, and model configuration
    checkpoint = {
        "model_name": model_name,
        "bid_zone": bid_zone,
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }

    # Save model configuration for reconstruction
    if quantiles is not None:
        checkpoint["quantiles"] = quantiles
    if num_features is not None:
        checkpoint["num_features"] = num_features

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)

    print(f"Saved checkpoint: {filename}")

    return path


def load_torch_model(
    file_name=None,
    model_name=None,
    dir_path="../checkpoints",
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

    model_class_name = checkpoint.get("model_class", "WindPowerQuantile")

    if model_class_name == "WindPowerQuantile":
        if "quantiles" not in checkpoint or "num_features" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model configuration (quantiles, num_features). "
                "Please save checkpoint with these parameters."
            )

        quantiles = checkpoint["quantiles"]
        num_features = checkpoint["num_features"]
        model = WindPowerQuantile(num_features, quantiles)
    elif model_class_name == "WindPowerModel_simple":
        num_features = checkpoint["num_features"]
        model = WindPowerModel_simple(num_features)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: {os.path.basename(path)}")
    if "bid_zone" in checkpoint:
        print(f"  Bid zone: {checkpoint['bid_zone']}")
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
        progress_bar=True,
    )
    mcmc.run(rng_key, X=X, Y=Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# Posterior predictive for basic BNN
def predict_bnn(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(model).get_trace(X=X, Y=None)
    return trace["Y"]["value"]


def run_svi(
    model, guide, rng_key, X_train, Y_train, optimizer=Adam, lr=1e-3, num_steps=5000
):
    optimizer = optimizer(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=8))

    svi_result = svi.run(rng_key, num_steps, X_train, Y_train, progress_bar=False)

    losses = svi_result.losses
    print(f"Final ELBO loss: {losses[-1]:.3f}")
    return svi_result


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
