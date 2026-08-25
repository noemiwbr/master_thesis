import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde, norm, percentileofscore

import numpyro.distributions as dist
from numpyro.infer import Predictive
import arviz as az

import torch

from data_processing.utils import QuantileDistribution

from training.evaluate import (
    evaluate_model,
    evaluate_numpyro_on_train_set,
)
import sys

sys.path.append("../src")
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import arviz as az


from data_processing.load import load_dataset
from data_processing.utils import load_torch_model, load_checkpoint
from models.build import build_model
from training.evaluate import (
    evaluate_model,
    evaluate_numpyro,
)


def plot_qr_prediction_ensemble(
    Y_true,
    Y_pred,
    title,
    timestamps,
    plot_interval=[0, 1000],
    cred_mass=0.95,
):

    ensemble = torch.stack(Y_pred, dim=0)  # [15, N, Q]
    ensemble_mean = torch.mean(ensemble, dim=0)  # [N, Q]

    quantiles = torch.linspace(0, 1, Y_pred[0].shape[-1])
    median_idx = torch.argmin(torch.abs(quantiles - 0.5)).item()
    lower_idx = torch.argmin(torch.abs(quantiles - ((1 - cred_mass) / 2))).item()
    upper_idx = torch.argmin(torch.abs(quantiles - (1 - (1 - cred_mass) / 2))).item()

    df_ensemble = pd.DataFrame(
        np.stack(
            [
                Y_true,
                ensemble_mean[:, lower_idx],
                ensemble_mean[:, upper_idx],
                ensemble_mean[:, median_idx],
            ]
        ).T,
        index=timestamps,
        columns=["Y_true", "hdi_lower", "hdi_upper", "median"],
    )

    df_member0 = pd.DataFrame(
        np.stack(
            [
                Y_true,
                Y_pred[0][:, lower_idx],
                Y_pred[0][:, upper_idx],
                Y_pred[0][:, median_idx],
            ]
        ).T,
        index=timestamps,
        columns=["Y_true", "hdi_lower", "hdi_upper", "median"],
    )

    full_index = pd.date_range(
        df_ensemble.index.min(), df_ensemble.index.max(), freq="h"
    )
    # Reindex and fill missing timestamps with NaN
    df_ensemble_full = df_ensemble.reindex(full_index)
    df_member0_full = df_member0.reindex(full_index)

    plt.figure(figsize=(7.2*2, 4), constrained_layout=True)

    label_size = 11
    tick_size = 10
    title_size = 12
    legend_size = 10

    plt.plot(
        df_member0_full.index[slice(*plot_interval)],
        df_member0_full["Y_true"][slice(*plot_interval)],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    plt.fill_between(
        df_member0_full.index[slice(*plot_interval)],
        df_member0_full["hdi_lower"][slice(*plot_interval)],
        df_member0_full["hdi_upper"][slice(*plot_interval)],
        color="lightblue",
        alpha=0.5,
        label=f"{int(cred_mass * 100)}% PI member 0",
    )
    plt.plot(
        df_member0_full.index[slice(*plot_interval)],
        df_member0_full["median"][slice(*plot_interval)],
        color="red",
        label="Median member 0",
        lw=2,
    )

    plt.fill_between(
        df_ensemble_full.index[slice(*plot_interval)],
        df_ensemble_full["hdi_lower"][slice(*plot_interval)],
        df_ensemble_full["hdi_upper"][slice(*plot_interval)],
        color="blue",
        alpha=0.1,
        label=f"{int(cred_mass * 100)}% PI across-members",
    )
    plt.plot(
        df_ensemble_full.index[slice(*plot_interval)],
        df_ensemble_full["median"][slice(*plot_interval)],
        color="blue",
        label="Mean Median across-members",
        lw=2,
    )

    plt.xlabel("Time", fontsize=label_size)
    plt.ylabel("Total power (MW)", fontsize=label_size)
    plt.title(title, fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size, ncol=2)

    return


def plot_qr_hdi_prediction_simple(
    Y_true,
    conf_interval,
    medians,
    plot_interval=[0, 1000],
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):

    df = pd.DataFrame(
        np.stack([Y_true, conf_interval[:, 0], conf_interval[:, 1], medians]).T,
        index=timestamps,
        columns=["Y_true", "hdi_lower", "hdi_upper", "median"],
    )
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    # Reindex and fill missing timestamps with NaN
    df_full = df.reindex(full_index)

    plt.figure(
        figsize=(15, 4),
        constrained_layout=True,
    )

    # Top plot
    plt.plot(
        df_full.index[slice(*plot_interval)],
        df_full["Y_true"][slice(*plot_interval)],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    plt.fill_between(
        df_full.index[slice(*plot_interval)],
        df_full["hdi_lower"][slice(*plot_interval)],
        df_full["hdi_upper"][slice(*plot_interval)],
        color="#A6CEE3",
        alpha=0.5,
        label=(
            f"{int(cred_mass * 100)}% HDI"
            if "BNN" in title
            else f"{int(cred_mass * 100)}% PI"
        ),
    )

    plt.plot(
        df_full.index[slice(*plot_interval)],
        df_full["median"][slice(*plot_interval)],
        color="red",
        label="Posterior median" if "BNN" in title else "Predicted median",
        lw=2,
    )

    plt.xlabel("Time")
    plt.ylabel("Total power (MW)")
    plt.title(title)
    plt.legend()


def plot_qr_hdi_prediction_ensemble_std(
    Y_true,
    ensemble_results,
    plot_interval=[0, 1000],
    cred_mass=0.95,
    dataset="forecast",
    title="QR Power Generation Prediction",
    timestamps=None,
):
    ens_df = ensemble_results.set_index(timestamps)
    ens_df["Y_true"] = Y_true

    full_index = pd.date_range(ens_df.index.min(), ens_df.index.max(), freq="h")

    # Reindex and fill missing timestamps with NaN
    ens_df_full = ens_df.reindex(full_index)

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 4),
        sharex=True,
        constrained_layout=True,
    )
    ax.set(
        ylabel="Total power (MW)",
        title=title,
    )

    if dataset == "forecast":
        for member in range(0, 15):
            alpha = 0.5 if member == 0 else 0.2

            ax.plot(
                ens_df_full.index[slice(*plot_interval)],
                ens_df_full[f"lower_{member}"][slice(*plot_interval)],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=(
                    f"95% HDI member-wise" if "BNN" in title else "95% PI member-wise"
                ),
            )
            ax.plot(
                ens_df_full.index[slice(*plot_interval)],
                ens_df_full[f"upper_{member}"][slice(*plot_interval)],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
            )
            ax.plot(
                ens_df_full.index[slice(*plot_interval)],
                ens_df_full[f"median_{member}"][slice(*plot_interval)],
                color="red",
                lw=2 if member == 0 else 0.8,
                alpha=alpha,
                linestyle="-",
                label=(
                    f"Posterior median member-wise"
                    if "BNN" in title
                    else "Predicted median member-wise"
                ),
            )

    ax.plot(
        ens_df_full.index[slice(*plot_interval)],
        ens_df_full["Y_true"][slice(*plot_interval)],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )
    ax.set_xlabel("Time")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return fig, ax


def extract_quantile_curves(
    Y_pred_quantiles,
    quantiles,
    median_q=0.5,
    lower_q=0.025,
    upper_q=0.975,
):
    def q_idx(q):
        idx = np.where(quantiles == q)[0]
        if len(idx) == 0:
            raise ValueError(f"Quantile {q} not found in quantiles.")
        return idx[0]

    median_idx = q_idx(median_q)
    lower_idx = q_idx(lower_q)
    upper_idx = q_idx(upper_q)

    median = Y_pred_quantiles[:, median_idx].numpy()
    lower = Y_pred_quantiles[:, lower_idx].numpy()
    upper = Y_pred_quantiles[:, upper_idx].numpy()

    return median, lower, upper


def plot_bnn_gamma_combined(
    checkpoint, model, data, data_cfg, plot_interval, split="test"
):

    # TODO: fix naming of data splits
    split_names = {"test": "validation", "train": "train", "eval": "test"}
    pp_samples_list = []
    samples = checkpoint["samples"]
    model_cfg = checkpoint["model_cfg"]

    predictive = Predictive(
        model,
        samples,
        return_sites=["mu", "alpha", "Y"],
    )

    for member in range(15):
        if model_cfg.backend == "numpyro":

            pp_samples = predictive(
                checkpoint["rng_key_predict"],
                X=data[f"X_{split}"][:, :, member],
                Y=None,
            )

            pp_samples_list.append(pp_samples["Y"])
        else:
            raise TypeError()

    pp_samples_all = np.concatenate(pp_samples_list, axis=0)
    hdi = az.hdi(np.expand_dims(pp_samples_all, axis=0), hdi_prob=0.95)
    median = np.percentile(pp_samples_all, 50, axis=0)

    pp_samples_0 = predictive(
        checkpoint["rng_key_predict"], X=data[f"X_{split}"][:, :, 0], Y=None
    )["Y"]
    hdi_0 = az.hdi(np.expand_dims(pp_samples_0, axis=0), hdi_prob=0.95)
    median_0 = np.percentile(pp_samples_0, 50, axis=0)

    if f"X_{split}_timestamps" in data.keys():
        timestamps = data[f"X_{split}_timestamps"]  # [slice(*plot_interval)]
    elif "index" in data.keys():
        timestamps = data["index"]
    else:
        timestamps = None

    plot_qr_hdi_prediction_ensemble_combined(
        Y_true=data[f"Y_{split}_gen"],
        conf_interval=[hdi, hdi_0],
        medians=[median, median_0],
        plot_interval=plot_interval,
        title=f"BNN: Aggregated ensemble member prediction ({split_names[split]} set, {model_cfg.data.bid_zone})",
        timestamps=timestamps,
    )
    return


def plot_quantile_calibration(
    Y_true,
    Y_pred_quantiles,
    quantiles,
    bz,
    model_name,
):

    Y_true = np.asarray(Y_true)
    Y_pred_quantiles = np.asarray(Y_pred_quantiles)
    quantiles = np.asarray(quantiles)

    emp_cov = np.mean(
        Y_true[:, None] <= Y_pred_quantiles,
        axis=0,
    )

    plt.figure(figsize=(6, 6))
    plt.plot(quantiles, emp_cov, "o-", label="Empirical")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    plt.xlabel("Nominal quantile τ")
    plt.ylabel("Empirical coverage")
    plt.title(f"{model_name}: Calibration plot ({bz})")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    return emp_cov


# Plot posterior weight distributions for selected BNN layers and compare them to the Normal prior
def plot_bnn_weight_posteriors(
    samples,
    layer_names=("w1", "b1", "w2", "b2"),
    prior_mean=0.0,
    prior_std=1.0,
    bins=80,
):

    layers = [(name, samples[name]) for name in layer_names]

    n = len(layers)
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes).ravel()

    x = np.linspace(-4, 4, 400)
    prior_pdf = np.exp(dist.Normal(prior_mean, prior_std).log_prob(x))

    for ax, (name, arr) in zip(axes, layers):
        post = np.asarray(arr).ravel()

        ax.hist(
            post,
            bins=bins,
            density=True,
            alpha=0.8,
            color="#A6CEE3",
            label="Posterior",
        )

        sns.kdeplot(
            post,
            ax=ax,
            fill=False,
            color="#1f77b4",
            alpha=1,
            label="KDE Posterior",
        )

        ax.plot(
            x,
            prior_pdf,
            "k--",
            lw=1.5,
            label=f"Prior N({prior_mean}, {prior_std})",
        )

        ax.set_title(f"Weight distribution: {name}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.legend()

    # Hide unused axes if any
    for ax in axes[len(layers) :]:
        ax.axis("off")

    plt.show()


# Plot prior vs posterior distributions of first-layer weights per feature, including an estimated KL divergence to the prior.
def plot_first_layer_feature_posteriors(
    samples,
    data,
    prior_mean=0.0,
    prior_std=1.0,
):
    feature_cols = data["feature_cols"]
    # Posterior samples: [S, F, H]
    w = samples["w1"]
    F = w.shape[1]

    x = np.linspace(-3, 3, 400)
    prior_pdf = np.exp(dist.Normal(prior_mean, prior_std).log_prob(x))
    ncols = 2
    nrows = int(np.ceil(F / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 3 * nrows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).flatten()

    for f, ax in enumerate(axes[:F]):
        post = np.asarray(w[:, f, :]).ravel()

        # Posterior KDE
        sns.kdeplot(
            post,
            ax=ax,
            color="#1f77b4",
            lw=2,
            label="KDE Posterior",
        )

        # Prior
        ax.plot(
            x,
            prior_pdf,
            color="black",
            ls="--",
            lw=1.2,
            label=f"Prior N({prior_mean}, {prior_std})",
        )

        # KL divergence estimate: D_KL(Posterior || Prior)
        kde = gaussian_kde(post)
        log_p_post = np.log(kde(post))
        log_p_prior = norm.logpdf(post, loc=prior_mean, scale=prior_std)
        kl_est = np.mean(log_p_post - log_p_prior)

        ax.text(
            0.05,
            0.92,
            f"KL ≈ {kl_est:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
        )

        ax.set_title(f"Feature: {feature_cols[f]}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.legend()

    # Hide unused subplots
    for ax in axes[F:]:
        ax.set_visible(False)

    plt.show()


def get_dist_theory(
    predictive,
    model_name,
    t,
):

    # --- Infer likelihood type ---
    model_name = model_name.lower()

    if "gamma" in model_name:
        alpha_samples = np.asarray(predictive["alpha"][:, t])
        mu_samples = np.asarray(predictive["mu"][:, t])

        alpha = alpha_samples.mean()
        mu = mu_samples.mean()
        rate = alpha / mu

        dist_theory = dist.Gamma(concentration=alpha, rate=rate)
        xlabel = "Power output (MW)"
        title = f"Posterior predictive vs. Gamma PDF (t={t})"
        label = f"Theoretical Gamma PDF\n(α={alpha:.2f}, μ={mu:.2f})"

    elif "beta" in model_name:
        alpha_samples = np.asarray(predictive["alpha"][:, t])
        beta_samples = np.asarray(predictive["beta"][:, t])

        alpha = alpha_samples.mean()
        beta = beta_samples.mean()

        dist_theory = dist.Beta(alpha, beta)
        xlabel = "Capacity factor"
        title = f"Posterior predictive vs. Beta PDF (t={t})"
        label = f"Theoretical Beta PDF\n(α={alpha:.2f}, β={beta:.2f})"

    else:
        raise ValueError("model_name must contain either 'gamma' or 'beta'")

    return dist_theory, xlabel, title, label


# Plot posterior predictive samples vs. theoretical likelihood PDF at a given timestep
def plot_posterior_predictive_vs_likelihood(
    predictive,
    model_name,
    t,
    bins=30,
):
    dist_theory, xlabel, title, label = get_dist_theory(
        predictive,
        model_name,
        t,
    )
    # --- Posterior predictive samples ---
    y_samples = np.asarray(predictive["Y"][:, t])

    # --- Evaluate theoretical PDF ---
    x = np.linspace(y_samples.min(), y_samples.max(), 200)

    pdf = np.asarray(np.exp(dist_theory.log_prob(np.asarray(x))))

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.hist(
        y_samples,
        bins=bins,
        density=True,
        alpha=0.6,
        color="#A6CEE3",
        label="Posterior predictive samples",
    )
    plt.plot(
        x,
        pdf,
        color="#D62728",
        lw=2.5,
        label=label,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# def plot_posterior_predictive_family(
#     predictive,
#     model_cfg,
#     data,
#     t,
#     hdi=None,
#     n_plot=200,
#     bins=30,
# ):

#     Y_true = (
#         data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]
#     )
#     model_name = model_cfg.name.lower()

#     # --- Posterior predictive samples ---
#     y_samples = np.asarray(predictive["Y"][:, t])

#     # --- X-range (robust to long tails) ---
#     xmax = float(np.quantile(y_samples, 0.995))
#     x = np.linspace(0, xmax, 400)

#     # --- Likelihood-specific parameters ---
#     if "gamma" in model_name:
#         alpha_samples = np.asarray(predictive["alpha"][:, t])
#         mu_samples = np.asarray(predictive["mu"][:, t])

#         alpha_mean = float(alpha_samples.mean())
#         mu_mean = float(mu_samples.mean())
#         rate_mean = alpha_mean / mu_mean

#         def pdf_fn(a, m):
#             return np.asarray(
#                 np.exp(dist.Gamma(concentration=a, rate=a / m).log_prob(np.asarray(x)))
#             )

#         pdf_mean = np.asarray(
#             np.exp(
#                 dist.Gamma(concentration=alpha_mean, rate=rate_mean).log_prob(
#                     np.asarray(x)
#                 )
#             )
#         )

#         xlabel = "Power output (MW)"
#         family_label = "Posterior Gamma samples"
#         mean_label = f"Mean Gamma PDF\n(α={alpha_mean:.2f}, μ={mu_mean:.2f})"

#     elif "beta" in model_name:
#         alpha_samples = np.asarray(predictive["alpha"][:, t])
#         beta_samples = np.asarray(predictive["beta"][:, t])

#         alpha_mean = float(alpha_samples.mean())
#         beta_mean = float(beta_samples.mean())

#         def pdf_fn(a, b):
#             return np.asarray(np.exp(dist.Beta(a, b).log_prob(np.asarray(x))))

#         pdf_mean = np.asarray(
#             np.exp(dist.Beta(alpha_mean, beta_mean).log_prob(np.asarray(x)))
#         )

#         xlabel = "Capacity factor"
#         family_label = "Posterior Beta samples"
#         mean_label = f"Mean Beta PDF\n(α={alpha_mean:.2f}, β={beta_mean:.2f})"

#     else:
#         raise ValueError("model_name must contain 'gamma' or 'beta'")

#     # --- Figure ---
#     fig, axes = plt.subplots(
#         1, 2, figsize=(12, 4), constrained_layout=True, sharex=True, sharey=True
#     )

#     # =========================
#     # LEFT: predictive samples
#     # =========================
#     kde = gaussian_kde(y_samples)
#     x_kde = np.linspace(0, xmax, 100)

#     if hdi is not None:
#         hdi_lower, hdi_upper = hdi[t]
#         axes[0].axvspan(
#             hdi_lower, hdi_upper, color="#D3D3D3", alpha=0.6, label="95% HDI samples"
#         )
#     axes[0].plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")
#     axes[0].hist(
#         y_samples,
#         bins=bins,
#         density=True,
#         alpha=0.8,
#         color="#A6CEE3",
#         label="Posterior predictive samples",
#     )

#     axes[0].set_title(f"Posterior predictive samples (t={t})")
#     axes[0].set_xlabel(xlabel)
#     axes[0].set_ylabel("Density")
#     axes[0].legend()


#     # =========================
#     # Shared decorations
#     # =========================
#     if Y_true is not None:
#         print("Y not none")
#         y_true = float(Y_true[t])

#         for ax in axes:
#             ax.axvline(y_true, color="red", lw=2, linestyle="--", label="True Y")
#             if "gamma" in model_name:
#                 ax.axvline(
#                     mu_mean, color="orange", lw=2, linestyle="--", label="Mean mu"
#                 )

#     # Harmonize y-limits
#     ymax = max(ax.get_ylim()[1] for ax in axes)
#     for ax in axes:
#         ax.set_ylim(0, ymax)
#         ax.set_xlim(0, xmax)
#     axes[0].legend()
#     axes[1].legend()
#     plt.show()


def plot_posterior_predictive_family(
    predictives,
    model_cfg,
    data,
    t,
    hdi=None,
    bins=30,
):
    y_true_all = (
        data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]
    )

    # Posterior predictive samples, handle different ensemble members
    y_samples = []
    mu_samples = []
    for predictive in predictives:
        y_samples.append(np.asarray(predictive["Y"][:, t]))
        mu_samples.append(np.asarray(predictive["mu"][:, t]))
    y_samples = np.concatenate(y_samples)
    mu_samples = np.concatenate(mu_samples)

    print(y_samples.shape)
    print(mu_samples.shape)

    mean_pred = float(mu_samples.mean())
    xlabel = "Power output (MW)"
    mean_label = r"Mean {\mu}"

    # Robust x-range
    xmax = float(np.quantile(y_samples, 0.995))
    x_kde = np.linspace(0, xmax, 200)

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # HDI
    if hdi is not None:
        hdi_lower, hdi_upper = hdi[t]
        ax.axvspan(
            hdi_lower,
            hdi_upper,
            color="#D3D3D3",
            alpha=0.6,
            label="95% HDI samples",
        )

    # Histogram and KDE
    ax.hist(
        y_samples,
        bins=bins,
        density=True,
        alpha=0.8,
        color="#A6CEE3",
        label="Posterior predictive samples",
    )

    kde = gaussian_kde(y_samples)
    ax.plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")

    # True value and posterior mean
    if y_true_all is not None:
        y_true = float(y_true_all[t])
        ax.axvline(y_true, color="red", lw=2, linestyle="--", label="True Y")

    ax.axvline(mean_pred, color="orange", lw=2, linestyle="--", label=mean_label)

    # Formatting
    ax.set_title(f"Posterior predictive samples (t={t})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(0, xmax)
    ax.legend()

    plt.show()


def plot_posterior_predictive_family_simple(
    predictive,
    model_cfg,
    data,
    t,
    hdi=None,
    bins=30,
):

    Y_true = (
        data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]
    )
    model_name = model_cfg.name.lower()

    # Posterior predictive samples
    y_samples = np.asarray(predictive[:, t])
    pred_median = np.median(y_samples)

    # X-range (robust to long tails)
    xmax = float(np.quantile(y_samples, 0.995))
    x_kde = np.linspace(0, xmax, 100)

    plt.figure(figsize=(6, 4))

    # KDE
    kde = gaussian_kde(y_samples)

    if hdi is not None:
        hdi_lower, hdi_upper = hdi[t]
        plt.axvspan(
            hdi_lower,
            hdi_upper,
            color="#D3D3D3",
            alpha=0.6,
            label="95% HDI samples",
        )

    plt.plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")

    plt.hist(
        y_samples,
        bins=bins,
        density=True,
        alpha=0.8,
        color="#A6CEE3",
        label="Posterior predictive samples",
    )

    plt.title(f"Posterior predictive samples (t={t})")
    plt.xlabel("Power output (MW)" if "gamma" in model_name else "Capacity factor")
    plt.ylabel("Density")

    # True value
    if Y_true is not None:
        y_true = float(Y_true[t])
        plt.axvline(y_true, color="red", lw=2, linestyle="--", label="True Y")

    plt.axvline(
        pred_median, color="blue", lw=2, linestyle="--", label="Median prediction"
    )

    plt.xlim(0, xmax)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_pit_from_quantiles(preds, y_true, quantiles):
    """
    preds [N, Q]
    y_true: [N]
    quantiles: [Q]
    """
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
    quantiles = np.asarray(quantiles)

    pits = np.empty(len(y_true))

    for t in range(len(y_true)):
        pits[t] = QuantileDistribution(quantiles, preds[t]).cdf(y_true[t])

    return pits


# Plot PIT (Probability Integral Transform) histogram
def plot_pit_histogram(
    preds,
    Y_true,
    model_cfg,
    bins=20,
    figsize=(5, 3),
):
    if model_cfg.backend == "numpyro":
        # Compute PIT values
        pits = np.array(
            [
                percentileofscore(
                    preds[:, t],
                    Y_true[t],
                    kind="mean",
                )
                / 100.0
                for t in range(preds.shape[1])
            ]
        )

    else:
        if model_cfg.params.quantiles is None:
            if model_cfg.params.num_quantiles is not None:
                quantiles = torch.linspace(0, 1, model_cfg.params.num_quantiles)
            else:
                quantiles = torch.linspace(0, 1, 51)
        else:
            quantiles = torch.tensor(model_cfg.params.quantiles)

        pits = compute_pit_from_quantiles(preds, Y_true, quantiles)

    model_name_for_title = "BNN" if model_cfg.backend == "numpyro" else "QRNN"
    # Plot histogram
    plt.figure(figsize=figsize)
    plt.hist(
        pits,
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.8,
        color="#A6CEE3",
        edgecolor="white",
        label=f"PIT histogram",
    )
    plt.axhline(
        1.0,
        color="k",
        ls="--",
        lw=1,
        label="Uniform reference",
    )

    plt.xlabel("PIT value")
    plt.ylabel("Density")
    plt.title(f"{model_name_for_title}: PIT histogram ({model_cfg.data.bid_zone})")
    plt.tight_layout()
    plt.legend()

    return pits


def plot_qr_hdi_prediction_ensemble_combined(
    Y_true,
    conf_interval,
    medians,
    plot_interval=[0, 1000],
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):

    conf_interval_all, conf_interval_0 = conf_interval
    medians_all, medians_0 = medians

    # Convert to dataframes, to reindex and fill missing timestamps with NaN for plotting
    df_0 = pd.DataFrame(
        np.stack([Y_true, conf_interval_0[:, 0], conf_interval_0[:, 1], medians_0]).T,
        index=timestamps,
        columns=["Y_true", "hdi_lower", "hdi_upper", "median"],
    )
    full_index_0 = pd.date_range(df_0.index.min(), df_0.index.max(), freq="h")
    df_full_0 = df_0.reindex(full_index_0)

    df_all = pd.DataFrame(
        np.stack(
            [Y_true, conf_interval_all[:, 0], conf_interval_all[:, 1], medians_all]
        ).T,
        index=timestamps,
        columns=["Y_true", "hdi_lower", "hdi_upper", "median"],
    )
    full_index_all = pd.date_range(df_all.index.min(), df_all.index.max(), freq="h")
    df_full_all = df_all.reindex(full_index_all)

    plt.figure(figsize=(7.2*2, 4))

    label_size = 11
    tick_size = 10
    title_size = 12
    legend_size = 10

    # Top plot
    plt.plot(
        df_full_0.index[slice(*plot_interval)],
        df_full_0["Y_true"][slice(*plot_interval)],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    plt.fill_between(
        df_full_0.index[slice(*plot_interval)],
        df_full_0["hdi_lower"][slice(*plot_interval)],
        df_full_0["hdi_upper"][slice(*plot_interval)],
        color="lightblue",
        alpha=0.5,
        label=f"{int(cred_mass * 100)}% HDI member 0",
    )
    plt.plot(
        df_full_0.index[slice(*plot_interval)],
        df_full_0["median"][slice(*plot_interval)],
        color="red",
        label="Posterior median member 0",
        lw=2,
    )

    plt.fill_between(
        df_full_all.index[slice(*plot_interval)],
        df_full_all["hdi_lower"][slice(*plot_interval)],
        df_full_all["hdi_upper"][slice(*plot_interval)],
        color="blue",
        alpha=0.1,
        label=f"{int(cred_mass * 100)}% HDI across-members",
    )
    plt.plot(
        df_full_all.index[slice(*plot_interval)],
        df_full_all["median"][slice(*plot_interval)],
        color="blue",
        label="Posterior median across-members",
        lw=2,
    )

    plt.ylabel("Total power (MW)", fontsize=label_size)
    plt.xlabel("Time", fontsize=label_size)
    plt.title(title, fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size, ncol=2)
    plt.tight_layout()

    return


# Combined calibration anaylsis
def calibration_diagnostics(data, preds, quantiles, model_name, bz):
    quantiles = np.asarray(quantiles)

    if model_name == "BNN":

        pred_quantiles = np.quantile(preds, quantiles, axis=0).T  # (N, Q)
        pit_source = preds  # (S, N)

    else:
        pred_quantiles = np.asarray(preds)
        pit_source = np.asarray(preds)

    Y_true = np.asarray(data["Y_test_gen"])

    # Calibration curve
    emp_cov = np.mean(Y_true[:, None] <= pred_quantiles, axis=0)

    # PIT
    if model_name == "BNN":
        pits = np.array(
            [
                percentileofscore(pit_source[:, t], Y_true[t], kind="mean") / 100.0
                for t in range(pit_source.shape[1])
            ]
        )
    else:
        pits = np.empty(len(Y_true))
        for t in range(len(Y_true)):
            pits[t] = QuantileDistribution(quantiles, pit_source[t]).cdf(Y_true[t])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    label_size = 11
    tick_size = 10
    title_size = 11
    legend_size = 9

    # --- Calibration plot ---
    axes[0].plot(quantiles, emp_cov, "o-", label="Empirical")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[0].set_xlabel("Nominal quantile τ", fontsize=label_size)
    axes[0].set_ylabel("Empirical coverage", fontsize=label_size)
    axes[0].set_title("Calibration Curve", fontsize=title_size)
    axes[0].tick_params(axis="both", labelsize=tick_size)
    axes[0].legend(fontsize=legend_size)
    axes[0].grid(True)

    # --- PIT histogram ---
    axes[1].hist(
        pits,
        bins=20,
        range=(0, 1),
        density=True,
        alpha=0.8,
        color="#A6CEE3",
        edgecolor="white",
        label="PIT histogram",
    )
    axes[1].axhline(
        1.0,
        color="k",
        ls="--",
        lw=1,
        label="Uniform reference",
    )
    axes[1].set_xlabel("PIT value", fontsize=label_size)
    axes[1].set_ylabel("Density", fontsize=label_size)
    axes[1].set_title("PIT Histogram", fontsize=title_size)
    axes[1].tick_params(axis="both", labelsize=tick_size)
    axes[1].legend(fontsize=legend_size)

    fig.suptitle(
        f"{model_name}: Probabilistic Calibration Diagnostics (validation set, {bz})",
        fontsize=12,
    )

    plt.tight_layout()


def agg_weights_plot(learned_weights, max_capacity, model_name, bz):

    softmax_caps = torch.softmax(torch.tensor(max_capacity), 0).numpy()
    x = np.arange(len(softmax_caps))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    ax.plot(
        x,
        softmax_caps,
        marker="o",
        lw=2,
        label="Softmax max capacity",
    )

    ax.plot(
        x,
        learned_weights,
        marker="o",
        lw=2,
        label="Learned weightings",
    )

    ax.set_xlabel("Wind park index")
    ax.set_ylabel("Weight")
    ax.set_title(f"{model_name}: wind park aggregation weights ({bz})")

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return


def window_plot(
    data, model_cfg, checkpoint, model, model_name_for_title, window_size=1080
):
    data_copy = copy.deepcopy(data)

    metrics_train_list = []
    window_labels = []
    for s in ["train", "test"]:
        length = len(data_copy[f"X_{s}"])
        for start in range(0, length, window_size):
            end = min(start + window_size, length)

            data_copy["X_train"] = data[f"X_{s}"][start:end]
            data_copy["Y_train_gen"] = data[f"Y_{s}_gen"][start:end]

            if model_cfg.backend == "numpyro":
                (
                    metrics_train,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = evaluate_numpyro_on_train_set(
                    model,
                    checkpoint["samples"],
                    checkpoint["rng_key_predict"],
                    data_copy,
                    model_cfg,
                )
            else:
                if model_cfg.params.num_quantiles is not None:
                    quantiles = torch.linspace(0, 1, model_cfg.params.num_quantiles)
                else:
                    quantiles = torch.linspace(0, 1, 51)

                # TODO: verwendet nur member 0!!
                metrics_train, _, _ = evaluate_model(
                    model, data_copy, quantiles, model_cfg
                )
                metrics_train["rmse_median"] = metrics_train["rmse_median_train"]
                metrics_train["crps"] = metrics_train["crps_train"]

            metrics_train_list.append(metrics_train)
            window_labels.append(f"{start}-{end}")

    # plotting
    rmse_vals = np.array([d["rmse_median"] for d in metrics_train_list])
    crps_vals = np.array([d["crps"] for d in metrics_train_list])

    plt.figure(figsize=(8, 4))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(rmse_vals))

    plt.plot(rmse_vals, marker="o", label="RMSE")
    plt.scatter(x[0], rmse_vals[0], s=100, zorder=3, color=default_colors[0])
    plt.scatter(x[-2], rmse_vals[-2], s=100, zorder=3, color=default_colors[0])

    plt.plot(crps_vals, marker="o", label="CRPS")
    plt.scatter(x[0], crps_vals[0], s=100, zorder=3, color=default_colors[1])
    plt.scatter(x[-2], crps_vals[-2], s=100, zorder=3, color=default_colors[1])

    plt.axvline(
        8, color="black", lw=1, linestyle="--", label="Next year reference line"
    )
    if len(rmse_vals) > 16:
        plt.axvline(
            16,
            color="black",
            lw=1,
            linestyle="--",
        )

    plt.axvspan(x[-2], x[-1], color="#A6CEE3", alpha=0.3, label="Validation")
    plt.xlabel("1.5 month windows")
    plt.ylabel("Value")
    plt.legend()
    plt.title(
        f"{model_name_for_title}: RMSE and CRPS across windows of 1.5 months ({model_cfg.data.bid_zone})"
    )
    plt.xticks(x[::2])
    plt.tight_layout()


def plot_qr_hdi_prediction_lines(
    data,
    median_list,
    pi_list,
    model_name_for_title,
    bz,
    plot_interval=[0, 1080],
):
    median_df = pd.DataFrame(
        np.column_stack(median_list),
        columns=[f"median_{i}" for i in range(len(median_list))],
    )

    pi_dict = {}
    for i, arr in enumerate(pi_list):
        pi_dict[f"lower_{i}"] = arr[0]
        pi_dict[f"upper_{i}"] = arr[1]

    pi_df = pd.DataFrame(pi_dict)

    ensemble_results = pd.concat([median_df, pi_df], axis=1)
    ensemble_results = ensemble_results.set_index(data["X_test_timestamps"])
    ensemble_results["Y_true"] = data["Y_test_gen"]

    # Reindex and fill missing timestamps with NaN
    full_index = pd.date_range(
        ensemble_results.index.min(), ensemble_results.index.max(), freq="h"
    )
    ens_df_full = ensemble_results.reindex(full_index)

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 4),
        sharex=True,
        constrained_layout=True,
    )
    ax.set(
        ylabel="Total power (MW)",
        title=f"{model_name_for_title}: Ensemble member predictions (validation set, {bz})",
    )

    for member in range(0, 15):
        alpha = 0.5 if member == 0 else 0.2

        ax.plot(
            ens_df_full.index[slice(*plot_interval)],
            ens_df_full[f"lower_{member}"][slice(*plot_interval)],
            color="blue",
            lw=1.5 if member == 0 else 1,
            alpha=alpha,
            linestyle="-",
            label=(
                f"95% HDI member-wise"
                if "BNN" == model_name_for_title
                else "95% PI member-wise"
            ),
        )
        ax.plot(
            ens_df_full.index[slice(*plot_interval)],
            ens_df_full[f"upper_{member}"][slice(*plot_interval)],
            color="blue",
            lw=1.5 if member == 0 else 1,
            alpha=alpha,
            linestyle="-",
        )
        ax.plot(
            ens_df_full.index[slice(*plot_interval)],
            ens_df_full[f"median_{member}"][slice(*plot_interval)],
            color="red",
            lw=2 if member == 0 else 0.8,
            alpha=alpha,
            linestyle="-",
            label=(
                f"Posterior median member-wise"
                if "BNN" == model_name_for_title
                else "Predicted median member-wise"
            ),
        )

    ax.plot(
        ens_df_full.index[slice(*plot_interval)],
        ens_df_full["Y_true"][slice(*plot_interval)],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )
    ax.set_xlabel("Time")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return fig, ax


def cdf_comparison_plot(
    t, model_cfg_BNN, model_cfg_QRNN, data_cfg, file_name_BNN, file_name_QRNN
):

    # DATA
    data_cfg.dataset = "forecast"
    data_cfg.ensemble = SimpleNamespace(members=None, variance=False)
    data = load_dataset(data_cfg)

    # EVALUATION of BNN and QRNN for all ensemble members
    y_pred_list_BNN = []
    y_pred_list_QRNN = []

    for member in range(15):

        quantiles = torch.linspace(0, 1, model_cfg_QRNN.params.num_quantiles)

        # QRNN
        model_QRNN, checkpoint_QRNN = load_torch_model(
            file_name=file_name_QRNN, quantiles=quantiles
        )

        _, y_pred_QRNN, _ = evaluate_model(
            model=model_QRNN,
            data=data,
            quantiles=quantiles,
            model_cfg=checkpoint_QRNN["model_cfg"],
            member=member,
        )
        y_pred_list_QRNN.append(y_pred_QRNN)

        # BNN
        checkpoint_BNN = load_checkpoint(file_name=file_name_BNN)
        model_BNN = build_model(checkpoint_BNN["model_cfg"], data)
        _, y_pred_BNN, _, _, _, _ = evaluate_numpyro(
            model=model_BNN,
            samples=checkpoint_BNN["samples"],
            rng_key_predict=checkpoint_BNN["rng_key_predict"],
            data=data,
            model_cfg=checkpoint_BNN["model_cfg"],
            member=member,
        )

        y_pred_list_BNN.append(y_pred_BNN)

    # AGGREGATE ensemble predictions
    # QRNN
    ensemble = torch.stack(y_pred_list_QRNN, dim=0)  # [M, N, Q]
    ensemble_mean = torch.mean(ensemble, dim=0)  # [N, Q]

    quantiles = torch.linspace(0, 1, model_cfg_QRNN.params.num_quantiles)

    q5_idx = torch.argmin(torch.abs(quantiles - 0.5)).item()
    q025_idx = torch.argmin(torch.abs(quantiles - 0.025)).item()
    q975_idx = torch.argmin(torch.abs(quantiles - 0.975)).item()

    qrnn_cdf_x = ensemble_mean[t].numpy()
    qrnn_cdf_y = quantiles.numpy()

    qrnn_median = ensemble_mean[t][q5_idx].item()
    qrnn_lower = ensemble_mean[t][q025_idx].item()
    qrnn_upper = ensemble_mean[t][q975_idx].item()

    # BNN
    y_samples = np.concatenate(y_pred_list_BNN, axis=0)[:, t]

    def compute_ecdf(samples):
        samples = np.sort(np.asarray(samples))
        n = len(samples)
        y = np.arange(1, n + 1) / n
        return samples, y

    x_ecdf, F_ecdf = compute_ecdf(y_samples)

    hdi_lower, hdi_upper = az.hdi(y_samples, hdi_prob=0.95)
    bnn_median = np.median(y_samples)

    # Observed value
    y_true = float(data["Y_test_gen"][t])

    # PLOT
    # Shared x-limits
    x_min = min(qrnn_cdf_x.min(), x_ecdf.min(), y_true, qrnn_lower, hdi_lower)
    x_max = max(qrnn_cdf_x.max(), x_ecdf.max(), y_true, qrnn_upper, hdi_upper)

    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

    # Left: QRNN
    axes[0].fill_betweenx(
        y=[0, 1],
        x1=qrnn_lower,
        x2=qrnn_upper,
        color="#A6CEE3",
        alpha=0.3,
        label="95% PI",
    )

    axes[0].plot(
        qrnn_cdf_x,
        qrnn_cdf_y,
        lw=1.5,
        color="black",
        label="CDF",
    )

    axes[0].axvline(
        qrnn_median,
        color="orange",
        lw=1,
        linestyle="--",
        label="Median",
    )

    axes[0].axvline(
        y_true,
        color="red",
        lw=1,
        linestyle="--",
        label="Observed y",
    )

    axes[0].set_title("QRNN")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("F(y)")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Right: BNN
    axes[1].fill_betweenx(
        y=[0, 1],
        x1=hdi_lower,
        x2=hdi_upper,
        color="#A6CEE3",
        alpha=0.3,
        label="95% HDI",
    )

    axes[1].step(
        x_ecdf,
        F_ecdf,
        where="post",
        lw=1.5,
        color="black",
        label="Empirical CDF",
    )

    axes[1].axvline(
        bnn_median,
        color="orange",
        lw=1,
        linestyle="--",
        label="Median",
    )

    axes[1].axvline(
        y_true,
        color="red",
        lw=1,
        linestyle="--",
        label="Observed y",
    )

    axes[1].set_title("BNN")
    axes[1].set_xlabel("y")
    axes[1].set_xlim(x_min, x_max)
    axes[1].legend()

    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    date = data["X_test_timestamps"][t]
    print(date)

    plt.suptitle(
        f"Comparison of CDF functions at {date.strftime("%Y-%m-%d")} (validation set, {model_cfg_BNN.data.bid_zone})",
        fontsize=16,
    )
    # plt.savefig(f"../plots/compare_cdfs_{model_cfg_BNN.data.bid_zone}.pdf")
