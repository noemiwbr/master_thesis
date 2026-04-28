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

from torch.utils.data import Dataset
import torch
import torch.optim as optim
from functools import partial
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam


class WindDataset(Dataset):
    def __init__(self, X, Y):
        # X: [T, S, F] or [T, S, E, F]
        # Y: [T]
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


from torch.utils.data import DataLoader


def make_loader(X, Y, batch_size=512, shuffle=True, num_workers=0):
    ds = WindDataset(X, Y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
    )


def train_loop(
    model,
    optimizer,
    criterion,
    loader,
    num_steps,
    log_every=250,
):
    model.train()
    losses = []
    step = 0

    while step < num_steps:
        for xb, yb in loader:
            if step >= num_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            preds = model(xb)
            loss = criterion(preds, yb)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % log_every == 0:
                print(f"Step {step:6d} | Loss: {loss.item():.4f}")

            step += 1

    return losses


def train_torch(model, data, quantiles, model_cfg, data_cfg):

    data = to_tensor(data)
    X = data["X_train"]
    Y = data["Y_train_gen"] if model_cfg.params.target == "gen" else data["Y_train_cf"]

    # Handle forecast vs nowcast (take member 0)
    if X.ndim == 4:
        X = X[
            :, :, 0
        ]  # [T, S, F], TODO: chooses first member, do we need this, training happens only on nowcast

    OPTIMIZER = {"adam": optim.Adam, "sgd": optim.SGD}
    optimizer = OPTIMIZER[model_cfg.training.optimizer](
        model.parameters(), lr=model_cfg.training.lr
    )

    LOSS = {
        "quantile_loss_weighted": quantile_loss_weighted,
        "quantile_loss": quantile_loss,
        "mse": torch.nn.MSELoss(),
    }

    loss_fn = LOSS[model_cfg.training.loss]
    if model_cfg.name != "WindPowerModel_simple":
        loss_fn = partial(loss_fn, quantiles=quantiles)

    criterion = lambda preds, target: loss_fn(preds, target)

    # DataLoader
    batch_size = getattr(model_cfg.training, "batch_size", 64)
    loader = make_loader(X, Y, batch_size=batch_size, shuffle=True)

    losses = train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loader=loader,
        num_steps=model_cfg.training.num_steps,
    )

    save_torch_checkpoint(
        model=model,
        model_cfg=model_cfg,
        optimizer=optimizer,
        num_features=X.shape[-1],
        dataset=data_cfg.dataset,
        losses=losses,
        num_parks=X.shape[-2],
    )

    return model, losses


def sample_numpyro(model, data, model_cfg, data_cfg, rng_key):

    X = data["X_train"]
    Y = data["Y_train_gen"] if model_cfg.params.target == "gen" else data["Y_train_cf"]

    # Handle forecast vs nowcast, TODO
    if X.ndim == 4:
        X = X[:, :, 0]  # first ensemble member
        print("use first ensemble member")

    args = {
        "num_warmup": model_cfg.args.num_warmup,
        "num_samples": model_cfg.args.num_samples,
        "num_chains": model_cfg.args.num_chains,
    }

    samples = run_inference_bnn(model, args, rng_key, X, Y)

    rng_key, rng_key_predict = random.split(
        rng_key
    )  # TODO: remove rng_key_pred from checkpoint

    save_checkpoint(
        samples,
        model_cfg,
        rng_key,
        rng_key_predict,
        dataset=data_cfg.dataset,
    )

    return samples, rng_key, rng_key_predict
