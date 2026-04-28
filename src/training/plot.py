import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from scipy.stats import gaussian_kde, norm, percentileofscore

import numpyro.distributions as dist
from numpyro.infer import Predictive
import arviz as az

import torch

from data_processing.utils import QuantileDistribution


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

    plt.figure(figsize=(15, 4), constrained_layout=True)
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

    plt.xlabel("Time")
    plt.ylabel("Total power (MW)")
    plt.title(title)
    plt.legend()

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

    keys = ["median", "hdi_upper", "hdi_lower"]
    colors = ["red", "lightblue", "lightblue"]
    alphas = [0.3, 1, 1]
    if dataset == "forecast":
        for member in range(0, 15):
            alpha = 0.5 if member == 0 else 0.2

            ax.plot(
                ens_df_full.index[slice(*plot_interval)],
                ens_df_full[f"hdi_lower_{member}"][slice(*plot_interval)],
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
                ens_df_full[f"hdi_upper_{member}"][slice(*plot_interval)],
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


def plot_qr_hdi_prediction_ensemble_std_bands(
    Y_true,
    ensemble_results,
    plot_interval=[0, 1000],
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):
    if timestamps is None:
        x = np.arange(plot_interval[0], plot_interval[1])
    else:
        x = timestamps[slice(*plot_interval)]

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

    keys = ["median", "hdi_upper", "hdi_lower"]
    colors = ["red", "lightblue", "lightblue"]
    alphas = [0.3, 1, 1]
    if "forecast" in title:
        for member in range(0, 15):
            alpha = 0.5 if member == 0 else 0.2

            ax.plot(
                x,
                ensemble_results[f"hdi_lower_{member}"][slice(*plot_interval)],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=f"95% HDI member 0",
            )
            ax.plot(
                x,
                ensemble_results[f"hdi_upper_{member}"][slice(*plot_interval)],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=f"95% HDI member 0",
            )
            ax.plot(
                x,
                ensemble_results[f"median_{member}"][slice(*plot_interval)],
                color="red",
                lw=2 if member == 0 else 0.8,
                alpha=alpha,
                linestyle="-",
                label=f"Posterior median member 0",
            )
            break  # only member 0

    for c, k, a in zip(colors, keys, alphas):
        ax.fill_between(
            x,
            ensemble_results["mean_" + k][slice(*plot_interval)]
            + ensemble_results["std_" + k][slice(*plot_interval)],
            ensemble_results["mean_" + k][slice(*plot_interval)]
            - ensemble_results["std_" + k][slice(*plot_interval)],
            color=c,
            alpha=a,
            label=f"mean and std of member's 95% HDI",
        )

    ax.plot(
        x,
        Y_true[slice(*plot_interval)],
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


def plot_posterior_predictive_family(
    predictive,
    model_cfg,
    data,
    t,
    hdi=None,
    n_plot=200,
    bins=30,
):

    Y_true = (
        data["Y_test_gen"] if model_cfg.params.target == "gen" else data["Y_test_cf"]
    )
    model_name = model_cfg.name.lower()

    # --- Posterior predictive samples ---
    y_samples = np.asarray(predictive["Y"][:, t])

    # --- X-range (robust to long tails) ---
    xmax = float(np.quantile(y_samples, 0.995))
    x = np.linspace(0, xmax, 400)

    # --- Likelihood-specific parameters ---
    if "gamma" in model_name:
        alpha_samples = np.asarray(predictive["alpha"][:, t])
        mu_samples = np.asarray(predictive["mu"][:, t])

        alpha_mean = float(alpha_samples.mean())
        mu_mean = float(mu_samples.mean())
        rate_mean = alpha_mean / mu_mean

        def pdf_fn(a, m):
            return np.asarray(
                np.exp(dist.Gamma(concentration=a, rate=a / m).log_prob(np.asarray(x)))
            )

        pdf_mean = np.asarray(
            np.exp(
                dist.Gamma(concentration=alpha_mean, rate=rate_mean).log_prob(
                    np.asarray(x)
                )
            )
        )

        xlabel = "Power output (MW)"
        family_label = "Posterior Gamma samples"
        mean_label = f"Mean Gamma PDF\n(α={alpha_mean:.2f}, μ={mu_mean:.2f})"

    elif "beta" in model_name:
        alpha_samples = np.asarray(predictive["alpha"][:, t])
        beta_samples = np.asarray(predictive["beta"][:, t])

        alpha_mean = float(alpha_samples.mean())
        beta_mean = float(beta_samples.mean())

        def pdf_fn(a, b):
            return np.asarray(np.exp(dist.Beta(a, b).log_prob(np.asarray(x))))

        pdf_mean = np.asarray(
            np.exp(dist.Beta(alpha_mean, beta_mean).log_prob(np.asarray(x)))
        )

        xlabel = "Capacity factor"
        family_label = "Posterior Beta samples"
        mean_label = f"Mean Beta PDF\n(α={alpha_mean:.2f}, β={beta_mean:.2f})"

    else:
        raise ValueError("model_name must contain 'gamma' or 'beta'")

    # --- Figure ---
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True, sharex=True, sharey=True
    )

    # =========================
    # LEFT: predictive samples
    # =========================
    kde = gaussian_kde(y_samples)
    x_kde = np.linspace(0, xmax, 100)

    if hdi is not None:
        hdi_lower, hdi_upper = hdi[t]
        axes[0].axvspan(
            hdi_lower, hdi_upper, color="#D3D3D3", alpha=0.6, label="95% HDI samples"
        )
    axes[0].plot(x_kde, kde(x_kde), color="black", lw=2, label="KDE")
    axes[0].hist(
        y_samples,
        bins=bins,
        density=True,
        alpha=0.8,
        color="#A6CEE3",
        label="Posterior predictive samples",
    )

    axes[0].set_title(f"Posterior predictive samples (t={t})")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # =========================
    # RIGHT: likelihood family
    # =========================
    n_plot = min(n_plot, len(alpha_samples))

    for i in range(n_plot):
        if "gamma" in model_name:
            pdf = pdf_fn(alpha_samples[i], mu_samples[i])
        else:
            pdf = pdf_fn(alpha_samples[i], beta_samples[i])

        if i == 0:
            axes[1].plot(x, pdf, color="#A6CEE3", alpha=0.3, label=family_label)
        else:
            axes[1].plot(x, pdf, color="#A6CEE3", alpha=0.3)

    axes[1].plot(
        x,
        pdf_mean,
        color="black",
        lw=2,
        label=mean_label,
    )

    axes[1].set_title(f"Posterior likelihood family (t={t})")
    axes[1].set_xlabel(xlabel)
    axes[1].legend()

    # =========================
    # Shared decorations
    # =========================
    if Y_true is not None:
        print("Y not none")
        y_true = float(Y_true[t])

        for ax in axes:
            ax.axvline(y_true, color="red", lw=2, linestyle="--", label="True Y")
            if "gamma" in model_name:
                ax.axvline(
                    mu_mean, color="orange", lw=2, linestyle="--", label="Mean mu"
                )

    # Harmonize y-limits
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, xmax)
    axes[0].legend()
    axes[1].legend()
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
