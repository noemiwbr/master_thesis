import torch.optim as optim
from training.losses import quantile_loss_weighted, quantile_loss
from data_processing.utils import (
    save_torch_checkpoint,
)
from jax import random
from data_processing.utils import (
    save_checkpoint,
    run_inference_bnn,
)
from data_processing.preprocess import (
    to_tensor,
)


def train_loop(
    model,
    optimizer,
    criterion,
    X,
    Y,
    num_steps,
    log_every=250,
):
    losses = []

    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, Y)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.4f}")

    return losses


def train_torch(model, data, quantiles, model_cfg, data_cfg):
    data = to_tensor(data, data_cfg)
    X = data["X_train"]
    Y = data["Y_train_gen"] if model_cfg.params.target == "gen" else data["Y_train_cf"]

    # Handle forecast vs nowcast
    if model_cfg.name == "WindPowerQuantile" and X.ndim == 4:
        X = X[:, :, 0]  # first ensemble member

    if model_cfg.name == "WindPowerQuantileEnsembles" and X.ndim != 4:
        raise ValueError(
            "WindPowerQuantileEnsembles requires ensemble input with shape "
            "[T, S, E, F], but got X.ndim = "
            f"{X.ndim} and shape {tuple(X.shape)}"
        )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg.training.lr,
    )

    # Loss
    LOSS = {
        "quantile_loss_weighted": quantile_loss_weighted,
        "quantile_loss": quantile_loss,
    }
    criterion = lambda preds, target: LOSS[model_cfg.training.loss](
        preds, target, quantiles=quantiles
    )

    # Training
    losses = train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        X=X,
        Y=Y,
        num_steps=model_cfg.training.num_steps,
    )

    # Save checkpoint
    save_torch_checkpoint(
        model=model,
        model_cfg=model_cfg,
        optimizer=optimizer,
        num_features=X.shape[-1],
        dataset=data_cfg.dataset,
        losses=losses,
    )

    return model, losses


def sample_numpyro(model, data, model_cfg, data_cfg):

    X = data["X_train"]
    Y = data["Y_train_gen"] if model_cfg.params.target == "gen" else data["Y_train_cf"]

    # Handle forecast vs nowcast
    if model_cfg.name in ["model_mw_gamma", "model_cf_beta"] and X.ndim == 4:
        X = X[:, :, 0]  # first ensemble member
        print("use first ensemble member")

    args = {
        "num_warmup": model_cfg.args.num_warmup,
        "num_samples": model_cfg.args.num_samples,
        "num_chains": model_cfg.args.num_chains,
    }

    rng_key, rng_key_predict = random.split(random.PRNGKey(0))

    samples = run_inference_bnn(model, args, rng_key, X, Y)

    save_checkpoint(
        samples,
        model_cfg,
        rng_key,
        rng_key_predict,
        dataset=data_cfg.dataset,
    )

    return samples, rng_key, rng_key_predict
