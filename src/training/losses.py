import torch


def quantile_loss(preds, target, quantiles=(0.025, 0.5, 0.975)):
    # preds: [T, Q], target: [T]
    diff = target.unsqueeze(-1) - preds  # [T, Q]
    quantiles = torch.tensor(quantiles).to(preds.device)  # [Q]

    # Pinball loss:
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff)
    return torch.mean(torch.mean(loss, dim=1))  # TODO sum or mean??


def quantile_loss_weighted(preds, target, quantiles):
    diff = target.unsqueeze(-1) - preds  # [T, Q]
    quantiles = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)  # [Q]

    # Pinball loss for each quantile
    loss_q = torch.maximum(quantiles * diff, (quantiles - 1) * diff)  # [T, Q]

    # Normalize weights to sum to 1
    weights = torch.sqrt(quantiles * (1 - quantiles))
    weights = weights / weights.sum()

    # Weighted average over quantiles
    weighted_loss = (loss_q * weights).mean()

    return weighted_loss
