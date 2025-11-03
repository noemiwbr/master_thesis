import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
from torch import nn


class WindPowerModel_simple(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        """
        x: [timesteps, num_sites, num_features]  (weather data)
        """
        y_site = self.shared_mlp(x).squeeze(-1) # [timesteps, num_sites, 1]
        y_sum = y_site.sum(dim=1) # [timesteps]
        
        return y_site, y_sum
    

class WindPowerModel(nn.Module):
    def __init__(self, num_sites, num_features, embed_dim, site_meta_dim):
        super().__init__()
        self.site_embeddings = nn.Embedding(num_sites, embed_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_features + embed_dim + site_meta_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
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
    

def model_mw_gamma(X, Y=None, nonlin=jnp.tanh, eps=1e-6):
    N, S, F = X.shape

    w1 = numpyro.sample("w1", dist.Normal(0.,1).expand([F,16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0.,1).expand([16]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0.,1).expand([16,8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0.,1).expand([8]).to_event(1))

    w3mu = numpyro.sample("w3mu", dist.Normal(0.,1).expand([8,1]).to_event(2))
    b3mu = numpyro.sample("b3mu", dist.Normal(0.,1).expand([1]).to_event(1))

    w3k  = numpyro.sample("w3k",  dist.Normal(0.,1).expand([8,1]).to_event(2))
    b3k  = numpyro.sample("b3k",  dist.Normal(0.,1).expand([1]).to_event(1))

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)
    mu = jax.nn.softplus((jnp.einsum("nsg,gd->nsd", z2, w3mu) + b3mu).squeeze(-1).sum(axis=1)) + eps
    alpha  = jax.nn.softplus((jnp.einsum("nsg,gd->nsd", z2, w3k)  + b3k ).squeeze(-1).sum(axis=1)) + 1e-3
    rate = alpha / mu
    
    # Likelihood: Y ~ Gamma(k, k/mu) 
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Gamma(concentration=alpha, rate=rate), obs=Y)


def model_cf_logitnormal(X, Y=None, nonlin=jnp.tanh, eps=1e-6):
    N, S, F = X.shape

    w1 = numpyro.sample("w1", dist.Normal(0.,1).expand([F,16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0.,1).expand([16]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0.,1).expand([16,8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0.,1).expand([8]).to_event(1))

    w3m = numpyro.sample("w3m", dist.Normal(0.,1).expand([8,1]).to_event(2))
    b3m = numpyro.sample("b3m", dist.Normal(0.,1).expand([1]).to_event(1))

    w3s = numpyro.sample("w3s", dist.Normal(0.,1).expand([8,1]).to_event(2))
    b3s = numpyro.sample("b3s", dist.Normal(0.,1).expand([1]).to_event(1))

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)
    mu = (jnp.einsum("nsg,gd->nsd", z2, w3m) + b3m).squeeze(-1).sum(axis=1)
    s = (jnp.einsum("nsg,gd->nsd", z2, w3s) + b3s).squeeze(-1).sum(axis=1)
    sigma = jax.nn.softplus(s) + 1e-3

    if Y is not None:
        CF_safe = Y*(1-2*eps)+eps
        y_star = jnp.log(CF_safe) - jnp.log1p(-CF_safe)  # logit

    # Likelihood: logit(Y) ~ Normal(mu, sigma)
    with numpyro.plate("data", N):
        numpyro.sample("y_star", dist.Normal(mu, sigma), obs=(y_star if Y is not None else None))


# Bayesian NN
def model(X, Y=None, nonlin=jax.nn.relu):
    """
    X.shape: [N, S, F] (timesteps, num_sites, num_features)
    Y.shape: [N], total power (scalar per timestamp)
    """
    N, S, F = X.shape

    # First layer weights
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([16]).to_event(1))

    # # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([8]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([8, 1]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([1]).to_event(1))

    # Observation noise
    alpha = numpyro.sample("alpha", dist.Exponential(1.0))  # shape > 0

    # Forward pass
    # [N, S, F] -> [N, S, D_H]
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    # [N, S, D_H] -> [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)
    # [N, S, D_H] ->[N, S, D_Y]
    out = jnp.einsum("nsg,gd->nsd", z2, w3) + b3
    # [N, S, D_Y] -> [N, S]
    out = out.squeeze(-1)  

    # Aggregate across sites
    y_pred = out.sum(axis=1)  # [N]

    beta = alpha / y_pred # rate (inverse scale)

    # Likelihood:  Y ~ Gamma(alpha, beta) 
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Gamma(alpha, beta), obs=Y) #non negative
        

def bnn_uninformed_embeddings(X, Y=None, D_emb=4):
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
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([16]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([8]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([8, 1]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([1]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsg,gd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)



def bnn_informed_embeddings(X, sites_meta, Y=None, D_emb=4):
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
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([16]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([8]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([8, 1]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([1]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsg,gd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)




def bnn_informed_embeddings(X, sites_meta, Y=None, D_emb=4):
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
    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F_aug, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([16]).to_event(1))

    # Second layer weights
    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([8]).to_event(1))

    # Final layer
    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([8, 1]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([1]).to_event(1))

    # Observation precision prior
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X_aug, w1) + b1)     # [N, S, D_H]
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)        # [N, S, D_H]
    out = jnp.einsum("nsg,gd->nsd", z2, w3) + b3               # [N, S, D_Y]
    out = out.squeeze(-1)                                      # [N, S]

    # Aggregate
    y_pred = out.sum(axis=1)  # [N]

    # Likelihood
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(y_pred, sigma_obs), obs=Y)



def quantile_gen_model(X, Y=None, quantiles=(0.025, 0.5, 0.975), nonlin=jnp.tanh):
    """
    X.shape: [N, S, F] (timestamps, num_sites, num_features)
    Y.shape: [N] (scalar per timestamp)
    """
    N, S, F = X.shape
    Q = len(quantiles)

    w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([F, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([16]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([8]).to_event(1))

    w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([8, Q]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([Q]).to_event(1))

    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)
    y_pred_q = (jnp.einsum("nsg,gq->nsq", z2, w3) + b3).sum(axis=1) # [N, Q]

    # Likelihood: Y ~ AsymmetricLaplaceQuantile(loc = y_pred_q, scale = 1.0, quantile = q)
    with numpyro.plate("data", N):
        for i, q in enumerate(quantiles):
            # Each quantile gets own ALQ likelihood centered at predicted mean
            numpyro.sample(
                f"Y_obs_q{int(q*100)}",
                dist.AsymmetricLaplaceQuantile(
                    loc=y_pred_q[:, i],
                    scale=1.0,
                    quantile=q
                ),
                obs=Y
                )
            

def quantile_cf_model(X, Y=None, quantiles=(0.025, 0.5, 0.975), nonlin=jnp.tanh, eps=1e-6):
    """
    Bayesian quantile model for capacity factor (bounded in (0,1)),
    using a logit transform and AsymmetricLaplaceQuantile likelihood.
    """
    N, S, F = X.shape
    Q = len(quantiles)

    w1 = numpyro.sample("w1", dist.Normal(0., 1.).expand([F, 16]).to_event(2))
    b1 = numpyro.sample("b1", dist.Normal(0., 1.).expand([16]).to_event(1))

    w2 = numpyro.sample("w2", dist.Normal(0., 1.).expand([16, 8]).to_event(2))
    b2 = numpyro.sample("b2", dist.Normal(0., 1.).expand([8]).to_event(1))

    w3 = numpyro.sample("w3", dist.Normal(0., 1.).expand([8, Q]).to_event(2))
    b3 = numpyro.sample("b3", dist.Normal(0., 1.).expand([Q]).to_event(1))

    # Forward pass
    z1 = nonlin(jnp.einsum("nsf,fh->nsh", X, w1) + b1)
    z2 = nonlin(jnp.einsum("nsh,hg->nsg", z1, w2) + b2)
    y_pred_q = (jnp.einsum("nsg,gq->nsq", z2, w3) + b3).sum(axis=1)  # [N, Q]

    # Logit transform of observed CF
    if Y is not None:
        CF_safe = Y * (1 - 2 * eps) + eps   # ensure in (0,1)
        y_star = jnp.log(CF_safe) - jnp.log1p(-CF_safe)  # logit(Y)
    else:
        y_star = None

    # Likelihood: logit(Y) ~ AsymmetricLaplaceQuantile(loc = y_pred_q, scale = 1.0, quantile = q)
    with numpyro.plate("data", N):
        for i, q in enumerate(quantiles):
            numpyro.sample(
                f"y_star_q{int(q*100)}",
                dist.AsymmetricLaplaceQuantile(
                    loc=y_pred_q[:, i],
                    scale=1.0,
                    quantile=q,
                ),
                obs=y_star
            )