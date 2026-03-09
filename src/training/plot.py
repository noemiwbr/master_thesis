import numpy as np
import matplotlib.pyplot as plt
import scipy
import numpyro.distributions as dist
import seaborn as sns
from scipy.stats import gaussian_kde, norm
import arviz as az
from scipy.stats import percentileofscore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_qr_hdi_prediction(
    Y_true,
    conf_interval,
    medians,
    n_plot=1000,
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

    if timestamps is None:
        timestamps = np.arange(n_plot)

    fig, axes = plt.subplots(
        nrows=3,
        figsize=(15, 5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    # Top plot
    ax = axes[0]
    ax.plot(
        df_full.index[:n_plot],
        df_full["Y_true"][:n_plot],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    ax.fill_between(
        df_full.index[:n_plot],
        df_full["hdi_lower"][:n_plot],
        df_full["hdi_upper"][:n_plot],
        color="#A6CEE3",
        alpha=0.5,
        label=f"{int(cred_mass * 100)}% credible interval",
    )

    ax.plot(
        df_full.index[:n_plot],
        df_full["median"][:n_plot],
        color="red",
        label="Posterior median",
        lw=2,
    )

    ax.set(
        ylabel="Total power",
        title=title,
    )
    ax.legend()

    # Bottom panel
    width = df_full["hdi_upper"][:n_plot] - df_full["hdi_lower"][:n_plot]
    width_min_max = (df_full["median"][:n_plot] - df_full["hdi_lower"][:n_plot]) / width
    ax2 = axes[1]
    ax2.plot(df_full.index[:n_plot], width_min_max, color="#A6CEE3")
    ax2.set(
        xlabel="Time",
        ylabel="Interval width",
        title=f"Min max normalized median",
    )

    ax3 = axes[2]
    ax3.plot(df_full.index[:n_plot], width, color="#A6CEE3")
    ax3.set(
        xlabel="Time",
        ylabel="Interval width",
        title=f"{int(cred_mass * 100)}% Credible Interval Width",
    )

    return fig, axes


def plot_qr_hdi_prediction_ensemble_combined(
    Y_true,
    conf_interval,
    medians,
    n_plot=1000,
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):
    if timestamps is None:
        timestamps = np.arange(n_plot)

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 3),
        sharex=True,
        constrained_layout=True,
    )

    # Top plot
    ax.plot(
        timestamps,
        Y_true[:n_plot],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    ax.fill_between(
        timestamps,
        conf_interval[1][:n_plot, 0],
        conf_interval[1][:n_plot, 1],
        color="lightblue",
        alpha=0.5,
        label=f"{int(cred_mass * 100)}% credible interval member 0",
    )
    ax.plot(
        timestamps,
        medians[1][:n_plot],
        color="red",
        label="Posterior median member 0",
        lw=2,
    )

    ax.fill_between(
        timestamps,
        conf_interval[0][:n_plot, 0],
        conf_interval[0][:n_plot, 1],
        color="blue",
        alpha=0.1,
        label=f"{int(cred_mass * 100)}% credible interval across-members",
    )
    ax.plot(
        timestamps,
        medians[0][:n_plot],
        color="blue",
        label="Posterior median across-members",
        lw=2,
    )

    ax.set(
        ylabel="Total power",
        xlabel="Time",
        title=title,
    )
    ax.legend()

    return fig, ax


def plot_qr_hdi_prediction_ensemble(
    Y_true,
    ensemble_results,
    n_plot=1000,
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):
    if timestamps is None:
        timestamps = np.arange(n_plot)

    fig, axes = plt.subplots(
        nrows=3,
        figsize=(15, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [4, 1, 1]},
    )

    # Top plot
    ax = axes[0]
    ax.plot(
        timestamps[:n_plot],
        Y_true[:n_plot],
        ".",
        color="black",
        label="Observed data",
        markersize=4,
    )

    ax.fill_between(
        timestamps[:n_plot],
        ensemble_results["hdi_lower_0"][:n_plot],
        ensemble_results["hdi_upper_0"][:n_plot],
        color="#A6CEE3",
        alpha=0.5,
        label=f"{int(cred_mass * 100)}% credible interval",
    )

    ax.set(
        ylabel="Total power",
        title=title,
    )

    keys = ["median", "hdi_upper", "hdi_lower"]
    colors = ["red", "blue", "blue"]
    if "forecast" in title:
        for member in range(0, 15):
            ax.plot(
                timestamps[:n_plot],
                ensemble_results[f"median_{member}"][:n_plot],
                color="orange",
                lw=0.5,
                alpha=0.8,
                linestyle="-",
                label=f"Median ensemble members",
            )

    for c, k in zip(colors, keys):
        ax.fill_between(
            timestamps[:n_plot],
            ensemble_results["mean_" + k][:n_plot]
            + ensemble_results["std_" + k][:n_plot],
            ensemble_results["mean_" + k][:n_plot]
            - ensemble_results["std_" + k][:n_plot],
            color=c,
            alpha=0.1,
            label=f"Std of 95% CI (ensemble members)",
        )

    ax.plot(
        timestamps[:n_plot],
        ensemble_results["median_0"][:n_plot],
        color="red",
        label="Posterior median",
        lw=2,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    # ax.legend()

    # Bottom panel
    width = (
        ensemble_results["hdi_upper_0"][:n_plot]
        - ensemble_results["hdi_lower_0"][:n_plot]
    )
    width_min_max = (
        ensemble_results["median_0"][:n_plot] - ensemble_results["hdi_lower_0"][:n_plot]
    ) / width
    ax2 = axes[1]
    ax2.plot(timestamps[:n_plot], width_min_max, color="#A6CEE3")
    ax2.set(
        xlabel="Time",
        ylabel="Interval width",
        title=f"Min max normalized median",
    )

    ax3 = axes[2]
    ax3.plot(timestamps[:n_plot], width, color="#A6CEE3")
    ax3.set(
        xlabel="Time",
        ylabel="Interval width",
        title=f"{int(cred_mass * 100)}% Credible Interval Width",
    )

    return fig, axes


def plot_qr_hdi_prediction_ensemble_outline(
    Y_true,
    ensemble_results,
    n_plot=1000,
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):
    if timestamps is None:
        x = np.arange(n_plot)
    else:
        x = timestamps[:n_plot]

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 4),
        sharex=True,
        constrained_layout=True,
    )

    ax.set(
        ylabel="Total power",
        title=title,
    )

    if "forecast" in title:
        for member in range(0, 15):
            alpha = 0.5 if member == 0 else 0.2
            ax.fill_between(
                x,
                ensemble_results[f"hdi_lower_{member}"][:n_plot],
                ensemble_results[f"hdi_upper_{member}"][:n_plot],
                color="lightblue",  # if member != 0 else "blue",
                alpha=1 / 15 if member != 0 else 1,
                label=f"95% HDI member-wise",
            )

            ax.plot(
                x,
                ensemble_results[f"median_{member}"][:n_plot],
                color="red",
                lw=2 if member == 0 else 0.8,
                alpha=alpha,
                linestyle="-",
                label=f"Posterior median member-wise",
            )

        hdi_lower_cols = [
            c for c in ensemble_results.columns if c.startswith("hdi_lower_")
        ]
        hdi_upper_cols = [
            c for c in ensemble_results.columns if c.startswith("hdi_upper_")
        ]
        min_vals = ensemble_results[hdi_lower_cols].min(axis=1)
        max_vals = ensemble_results[hdi_upper_cols].max(axis=1)
        ax.plot(
            x,
            min_vals[:n_plot],
            color="lightblue",
            lw=1,
            alpha=1,
            label="Outer bounds across member's HDIs",
        )
        ax.plot(
            x,
            max_vals[:n_plot],
            color="lightblue",
            lw=1,
            alpha=1,
        )

    ax.plot(
        x,
        Y_true[:n_plot],
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


def plot_qr_hdi_prediction_ensemble_std(
    Y_true,
    ensemble_results,
    n_plot=1000,
    cred_mass=0.95,
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
        ylabel="Total power",
        title=title,
    )

    keys = ["median", "hdi_upper", "hdi_lower"]
    colors = ["red", "lightblue", "lightblue"]
    alphas = [0.3, 1, 1]
    if "forecast" in title:
        for member in range(0, 15):
            alpha = 0.5 if member == 0 else 0.2

            ax.plot(
                ens_df_full.index[:n_plot],
                ens_df_full[f"hdi_lower_{member}"][:n_plot],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=f"95% HDI member-wise",
            )
            ax.plot(
                ens_df_full.index[:n_plot],
                ens_df_full[f"hdi_upper_{member}"][:n_plot],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
            )
            ax.plot(
                ens_df_full.index[:n_plot],
                ens_df_full[f"median_{member}"][:n_plot],
                color="red",
                lw=2 if member == 0 else 0.8,
                alpha=alpha,
                linestyle="-",
                label=f"Posterior median member-wise",
            )

    ax.plot(
        ens_df_full.index[:n_plot],
        ens_df_full["Y_true"][:n_plot],
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
    n_plot=1000,
    cred_mass=0.95,
    title="QR Power Generation Prediction",
    timestamps=None,
):
    if timestamps is None:
        x = np.arange(n_plot)
    else:
        x = timestamps[:n_plot]

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 4),
        sharex=True,
        constrained_layout=True,
    )

    ax.set(
        ylabel="Total power",
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
                ensemble_results[f"hdi_lower_{member}"][:n_plot],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=f"95% HDI member 0",
            )
            ax.plot(
                x,
                ensemble_results[f"hdi_upper_{member}"][:n_plot],
                color="blue",
                lw=1.5 if member == 0 else 1,
                alpha=alpha,
                linestyle="-",
                label=f"95% HDI member 0",
            )
            ax.plot(
                x,
                ensemble_results[f"median_{member}"][:n_plot],
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
            ensemble_results["mean_" + k][:n_plot]
            + ensemble_results["std_" + k][:n_plot],
            ensemble_results["mean_" + k][:n_plot]
            - ensemble_results["std_" + k][:n_plot],
            color=c,
            alpha=a,
            label=f"mean and std of member's 95% HDI",
        )

    ax.plot(
        x,
        Y_true[:n_plot],
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
    lower_q=0.02,
    upper_q=0.98,
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


import torch
from numpyro.infer import Predictive


def plot_bnn_gamma_combined(checkpoint, model, data, data_cfg, n_plot, split="test"):

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
        timestamps = data[f"X_{split}_timestamps"][:n_plot]
    elif "index" in data.keys():
        timestamps = data["index"]
    else:
        timestamps = None

    fig, ax = plot_qr_hdi_prediction_ensemble_combined(
        Y_true=data[f"Y_{split}_gen"],
        conf_interval=[hdi, hdi_0],
        medians=[median, median_0],
        n_plot=n_plot,
        title=f"{model_cfg.name} {model_cfg.data.bid_zone} {data_cfg.dataset} {split_names[split]}",
        timestamps=timestamps,
    )
    return fig, ax


# def plot_quantile_calibration(
#     Y_true,
#     Y_pred_quantiles,
#     quantiles,
# ):
#     # empirical coverage: P(Y <= Q_tau)
#     emp_cov = torch.mean(
#         (Y_true[:, None] <= Y_pred_quantiles).float(),
#         dim=0,
#     )
#     q_np = quantiles.numpy()
#     emp_np = emp_cov.numpy()

#     plt.figure(figsize=(6, 6))
#     plt.plot(q_np, emp_np, "o-", label="Empirical")
#     plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
#     plt.xlabel("Nominal quantile τ")
#     plt.ylabel("Empirical coverage")
#     plt.title("Calibration plot (quantile reliability)")
#     plt.legend()
#     plt.grid(True)

#     return np.array(emp_cov)


def plot_quantile_calibration(
    Y_true,
    Y_pred_quantiles,
    quantiles,
    bz,
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
    plt.title(f"Calibration plot ({bz})")
    plt.legend()
    plt.grid(True)

    return emp_cov


# Plot residual histogram for BNN predictions and report skewness.
def plot_residuals(
    y_pred_median,
    Y_true,
    bins=50,
):
    # Residuals
    resid = Y_true - y_pred_median

    # Convert once for stats / plotting
    resid_np = np.asarray(resid)

    print("skew:", scipy.stats.skew(resid_np))

    plt.hist(resid_np, bins=bins)
    plt.xlabel("Residual (Y - E[Y|X])")
    plt.ylabel("Frequency")
    plt.title("BNN residuals (predictive mean)")
    plt.grid(True)
    plt.show()

    return resid_np


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


import numpyro.distributions as dist
from scipy.stats import gaussian_kde


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


# Plot stacked HDI intervals for posterior predictive samples
def plot_stacked_hdi_intervals(
    preds,
    Y_true,
    n_plot=1000,
    hdi_probs=None,
    timestamps=None,
    figsize=(15, 6),
):

    if hdi_probs is None:
        hdi_probs = np.arange(0.05, 1.0, 0.05)

    # Compute HDIs
    preds_hdi = np.expand_dims(preds, axis=0)
    hdis = {float(p): az.hdi(preds_hdi, hdi_prob=float(p)) for p in hdi_probs}

    plt.figure(figsize=figsize)
    # Time axis
    if timestamps is None:
        X = np.arange(n_plot)
        plt.xlim(0, min(int(n_plot), len(X) - 1))
    else:
        X = timestamps[:n_plot]
        plt.xlim(X[0], X[min(int(n_plot), len(X) - 1)])
        plt.gcf().autofmt_xdate()

    # Median prediction
    median = np.percentile(preds, 50, axis=0)

    max_p = max(hdi_probs)
    for p in hdi_probs:
        hdi = hdis[p][:n_plot]
        alpha = 0.2 * float(p)
        label = f"{int(p * 100)}% HDI" if p == max_p else None

        plt.fill_between(
            X,
            hdi[:, 0],
            hdi[:, 1],
            color="C0",
            alpha=alpha,
            lw=0,
            label=label,
        )

    plt.plot(X, median[:n_plot], color="C0", lw=2, label="Median prediction")
    plt.plot(X, Y_true[:n_plot], ".", color="black", label="Observed Y")

    plt.xlabel("Time")
    plt.ylabel("Predicted Total Power")
    plt.title("Stacked HDI intervals")
    plt.legend()
    plt.show()


# Plot PIT (Probability Integral Transform) histogram
def plot_pit_histogram(
    preds,
    Y_true,
    bz,
    bins=20,
    figsize=(5, 3),
    print_mean=True,
):

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

    if print_mean:
        pits_mean = pits.mean()
        print(f"PIT mean: {pits_mean:.4f} (should be ~1 if calibrated)")

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
        label=f"PIT histogram ({bz})",
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
    plt.title("PIT histogram")
    plt.legend()
    plt.show()

    return pits


def evaluate_and_plot_qr_ensemble_medians(
    model, data, quantiles, n_members, n_plot=1000
):
    """
    Evaluate a quantile regression model on ensemble members:
    - compute predictive medians
    - compute RMSE per ensemble member
    - plot medians vs. observations
    """
    highlight_member = 0
    X_test = data["X_test"]
    Y_true = data["Y_test_gen"]
    model.eval()

    quantiles = np.asarray(quantiles)
    median_idx = np.where(quantiles == 0.5)[0][0]

    if isinstance(Y_true, torch.Tensor):
        Y_true_np = Y_true.cpu().numpy()
    else:
        Y_true_np = np.asarray(Y_true)

    medians = []
    rmses = []

    with torch.no_grad():
        for m in range(n_members):
            X_test_m = torch.tensor(
                X_test[:, :, m, :],
                dtype=torch.float32,
            )

            y_pred = model(X_test_m)  # (T, Q)
            median_m = y_pred[:, median_idx].cpu().numpy()
            medians.append(median_m)
            rmse_m = np.sqrt(np.mean((Y_true_np - median_m) ** 2))
            rmses.append(rmse_m)

            print(f"Member {m} RMSE: {rmse_m:.4f}")

    medians = np.stack(medians)  # (M, T)
    rmse_mean = float(np.mean(rmses))
    print(f"\nAverage RMSE: {rmse_mean:.4f}")

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(15, 6))

    for m in range(n_members):
        if m == highlight_member:
            continue
        plt.plot(
            medians[m, :n_plot],
            alpha=0.8,
            linewidth=1,
            label=f"Ens {m}",
        )

    plt.plot(
        medians[highlight_member, :n_plot],
        alpha=1,
        linewidth=2,
        label=f"Ens {highlight_member}",
    )

    plt.plot(
        Y_true_np[:n_plot],
        ".",
        color="black",
        label="Observed",
    )

    plt.title(
        "Predictive medians for all ensemble members\n"
        f"(model trained on member {highlight_member})"
    )
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()


# Plot ensemble variance per lead time for each feature.
def plot_variance_per_feature(var_by_feature):
    for feat, series in var_by_feature.items():
        plt.figure(figsize=(6, 4))
        plt.plot(series.index, series.values, label=feat)
        plt.xlabel("Lead time (hours)")
        plt.ylabel("Variance across ensemble members")
        plt.title(f"Ensemble variance per lead time: {feat}")
        plt.legend()
        plt.grid(True)
        plt.show()


# Plot average ensemble variance across all features.
def plot_average_feature_variance(var_overall_features):
    plt.figure(figsize=(6, 4))
    plt.plot(var_overall_features.mean(axis=1))
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Average ensemble variance")
    plt.title("Average ensemble variance across all features")
    plt.grid(True)
    plt.show()


# Plot overall ensemble variance across all features and members.
def plot_overall_ensemble_variance(overall_variance):
    plt.figure(figsize=(10, 5))
    plt.plot(overall_variance.index, overall_variance.values)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("Overall ensemble variance")
    plt.title("Overall variance across all features and ensemble members")
    plt.grid(True)
    plt.show()
