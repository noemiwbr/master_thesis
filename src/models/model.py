import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
from torch import nn
import torch.nn.functional as F


class WindPower_QRNN_weighted(nn.Module):
    def __init__(
        self,
        num_parks,
        num_features,
        quantiles=torch.tensor([0.025, 0.5, 0.975]),
        target="gen",
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_parks = num_parks

        Q = len(quantiles)
        assert 0.5 in quantiles, "Median must be included in the quantiles"

        assert target == "gen" or target == "cf", "Taget must be either gen or cf"
        function_mapping = {"gen": nn.Softplus(), "cf": nn.Sigmoid()}
        self.monotonic_function = function_mapping[target]

        # Indices
        self.mid_idx = (self.quantiles == 0.5).nonzero()[0].item()
        self.left_count = self.mid_idx
        self.right_count = Q - self.mid_idx - 1

        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Softplus(),
            nn.Linear(16, 8),
            nn.Softplus(),
        )

        # raw logits -> softmax -> non-negative, sum to 1
        self.agg_logits = nn.Parameter(torch.ones(num_parks))

        # Quantile heads
        self.median_head = nn.Linear(8, 1)
        self.left_head = nn.Linear(8, self.left_count)  # for tau < 0.5
        self.right_head = nn.Linear(8, self.right_count)  # for tau > 0.5

    def forward(self, x):
        """
        x: [timesteps, num_sites, num_features]
        """
        y_site_q = self.shared_mlp(x)  # [T, S, 8]

        w = F.softmax(self.agg_logits, dim=0)  # [S]
        w_view = (
            w.view(1, -1, 1) if (len(x.shape) == 3) else w.view(1, -1, 1, 1)
        )  # TODO, müsste ja immer shape 3 sein
        y_sum_q = (y_site_q * w_view).sum(dim=1)

        # Median
        m = self.median_head(y_sum_q)

        # Positive distances
        d_left = self.monotonic_function(self.left_head(y_sum_q))  # [T, left_count]
        d_right = self.monotonic_function(self.right_head(y_sum_q))  # [T, right_count]

        # quantiles left of median (descending)
        q_left = m - torch.flip(torch.cumsum(d_left, dim=-1), dims=[-1])

        # quantiles right of median (ascending)
        q_right = m + torch.cumsum(d_right, dim=-1)

        # concat: [T, K]
        y_q = self.monotonic_function(torch.cat([q_left, m, q_right], dim=-1))

        return y_q


def windpower_BNN_gamma_weighted(X, Y=None, nonlin=jax.nn.softplus, eps=1e-6):
    N, S, F = X.shape

    w1 = numpyro.sample("w1", dist.Normal(0.0, 1).expand([F, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 1).expand([16]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0.0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1).expand([8]).to_event(1))

    w3mu = numpyro.sample("w3mu", dist.Normal(0.0, 1).expand([8, 1]).to_event(2))
    b3mu = numpyro.sample("b3mu", dist.Normal(0.0, 1).expand([1]).to_event(1))

    w3k = numpyro.sample("w3k", dist.Normal(0.0, 1).expand([8, 1]).to_event(2))
    b3k = numpyro.sample("b3k", dist.Normal(0.0, 1).expand([1]).to_event(1))

    # Learned aggregation weights for parks
    agg_logits = numpyro.sample(
        "agg_logits", dist.Dirichlet(jnp.ones(S))
    )  # Dirichlet prior
    agg_w = numpyro.deterministic("agg_w", agg_logits)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)

    mu_raw = (jnp.einsum("nsg,gd->nsd", z2, w3mu) + b3mu).squeeze(-1)
    alpha_raw = (jnp.einsum("nsg,gd->nsd", z2, w3k) + b3k).squeeze(-1)

    # Weighted aggregation
    mu_agg = jnp.sum(mu_raw * agg_w[None, :], axis=1)  # (N,)
    alpha_agg = jnp.sum(alpha_raw * agg_w[None, :], axis=1)  # (N,)

    mu = numpyro.deterministic("mu", jax.nn.softplus(mu_agg) + eps)
    alpha = numpyro.deterministic("alpha", jax.nn.softplus(alpha_agg) + 1e-3)

    rate = alpha / mu

    # Likelihood: Y ~ Gamma(alpha, alpha/mu)
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Gamma(concentration=alpha, rate=rate), obs=Y)
