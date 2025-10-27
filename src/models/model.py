import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
from torch import nn


class WindPowerModel(nn.Module):
    def __init__(self, num_sites, num_features, embed_dim, site_meta_dim):
        super().__init__()
        self.site_embeddings = nn.Embedding(num_sites, embed_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features + embed_dim + site_meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, site_meta):
        """
        x: [timesteps, num_sites, num_features]  (weather data)
        site_meta: [num_sites, site_meta_dim] (known info per site, e.g. capacity)
        """
        batch_size, num_sites, _ = x.shape
        site_ids = torch.arange(num_sites, device=x.device)
        site_embeds = self.site_embeddings(site_ids) # [num_sites, embed_dim]

        # Combine site meta + embedding
        site_info = torch.cat([site_embeds, site_meta], dim=-1) # [num_sites, embed_dim+site_meta_dim]
        site_info = site_info.unsqueeze(0).expand(batch_size, -1, -1)

        inputs = torch.cat([x, site_info], dim=-1) # [timesteps, num_sites, num_features+embed_dim+site_meta_dim]
        y_site = self.shared_mlp(inputs).squeeze(-1) # [timesteps, num_sites, 1]

        y_sum = y_site.sum(dim=1) # [timesteps]
        
        return y_site, y_sum
    
# Nonlinearity
def nonlin(x):
    return jnp.tanh(x)


# Bayesian NN
def model(X, Y=None, D_H=32, D_Y=1):
    """
    X.shape: [N, S, F] (timesteps, num_sites, num_features)
    Y.shape: [N], total power (scalar per timestamp)
    """
    N, S, F = X.shape

    # First layer weights
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F, D_H]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([D_H, D_H]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([D_H, D_Y]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([D_Y]).to_event(1))

    # Observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    # [N, S, F] -> [N, S, D_H]
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    # [N, S, D_H] -> [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hh->nsh", z1, w2) + b2)
    # [N, S, D_H] ->[N, S, D_Y]
    out = jnp.einsum("nsh,hd->nsd", z2, w3) + b3
    # [N, S, D_Y] -> [N, S]
    out = out.squeeze(-1)  

    # Aggregate across sites
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y) # TODO Log Normal, Chi, Gamma, non negative


def bnn_uninformed_embeddings(X, Y=None, D_H=32, D_Y=1, D_emb=4):
    """
    X.shape: [N, S, F]
    Y.shape: [N]
    D_emb: embedding dimension per site
    """
    N, S, F = X.shape

    # One embedding vector per site, [S, D_emb]
    site_emb = numpyro.sample("site_emb", dist.Normal(0, 1).expand([S, D_emb]).to_event(2))
    # Broadcast embeddings across batch dimension -> [N, S, D_emb]
    site_emb_batched = jnp.broadcast_to(site_emb[None, :, :], (N, S, D_emb))

    # Concatenate embeddings to input features -> [N, S, F + D_emb]
    X_aug = jnp.concatenate([X, site_emb_batched], axis=-1)
    F_aug = F + D_emb

    # First layer weights
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, D_H]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([D_H, D_H]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([D_H, D_Y]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([D_Y]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hh->nsh", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsh,hd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)



def bnn_informed_embeddings(X, sites_meta, Y=None, D_H=32, D_Y=1, D_emb=4):
    """
    X.shape: [N, S, F]
    site_meta.shape: [S, D_meta]
    Y.shape: [N]
    D_emb: embedding dimension per site
    """
    N, S, F = X.shape
    D_meta = sites_meta.shape[-1]

    # Bayesian linear layer: sites_meta -> site_emb
    w_meta = numpyro.sample("w_meta", dist.Normal(0, 1).expand([D_meta, D_emb]).to_event(2))
    b_meta = numpyro.sample("b_meta", dist.Normal(0, 1).expand([D_emb]).to_event(1))

    sites_emb = jnp.dot(sites_meta, w_meta) + b_meta  # [S, D_emb]
    sites_emb_batched = jnp.broadcast_to(sites_emb[None, :, :], (N, S, D_emb))

    # Concatenate embeddings to input features -> [N, S, F + D_emb]
    X_aug = jnp.concatenate([X, sites_emb_batched], axis=-1)
    F_aug = F + D_emb

    # First layer weights
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, D_H]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([D_H, D_H]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([D_H, D_Y]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([D_Y]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hh->nsh", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsh,hd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)




def bnn_informed_embeddings(X, sites_meta, Y=None, D_H=32, D_Y=1, D_emb=4):
    """
    X.shape: [N, S, F]
    site_meta.shape: [S, D_meta]
    Y.shape: [N]
    D_emb: embedding dimension per site
    """
    N, S, F = X.shape
    D_meta = sites_meta.shape[-1]

    # Bayesian linear layer: sites_meta -> site_emb
    w_meta = numpyro.sample("w_meta", dist.Normal(0, 1).expand([D_meta, D_emb]).to_event(2))
    b_meta = numpyro.sample("b_meta", dist.Normal(0, 1).expand([D_emb]).to_event(1))

    sites_emb = jnp.dot(sites_meta, w_meta) + b_meta  # [S, D_emb]
    sites_emb_batched = jnp.broadcast_to(sites_emb[None, :, :], (N, S, D_emb))

    # Concatenate embeddings to input features -> [N, S, F + D_emb]
    X_aug = jnp.concatenate([X, sites_emb_batched], axis=-1)
    F_aug = F + D_emb

    # First layer weights
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, D_H]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([D_H, D_H]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([D_H]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([D_H, D_Y]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([D_Y]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hh->nsh", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsh,hd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)



def quantile_bnn_model(X, Y=None, D_H=32, quantiles=(0.05, 0.5, 0.95), nonlin=jnp.tanh):
    """
    X.shape: [N, S, F] (timestamps, num_sites, num_features)
    Y.shape: [N] (scalar per timestamp)
    """
    N, S, F = X.shape
    Q = len(quantiles)

    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F, D_H]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([D_H]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([D_H, D_H]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([D_H]).to_event(1))

    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([D_H, Q]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([Q]).to_event(1))

    scale = numpyro.sample("scale", dist.Exponential(1.0))

    # [N, S, F] -> [N, S, D_H]
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    # [N, S, D_H] -> [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hh->nsh", z1, w2) + b2)
    # [N, S, D_H] -> [N, S, Q]
    out = jnp.einsum("nsh,hq->nsq", z2, w3) + b3
    # Aggregate across sites: [N, S, Q] -> [N, Q]
    y_pred_q = out.sum(axis=1)

    # Likelihood (Asymmetric Laplace for each quantile)
    with numpyro.plate("data", N):
        for i, q in enumerate(quantiles):
            numpyro.sample(
                f"Y_q{q:.2f}",
                dist.AsymmetricLaplace(loc=y_pred_q[:, i], scale=scale, asymmetry=q),
                obs=Y
            )
