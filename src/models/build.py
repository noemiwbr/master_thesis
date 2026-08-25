import sys

sys.path.append("../src")

from models.model import (
    WindPower_QRNN_weighted,
    windpower_BNN_gamma_weighted,
)
import numpy as np

# Registries
TORCH_MODELS = {
    "WindPower_QRNN_weighted": WindPower_QRNN_weighted,
}

NUMPYRO_MODELS = {
    "windpower_BNN_gamma_weighted": windpower_BNN_gamma_weighted,
    "model_mw_gamma_extended": windpower_BNN_gamma_weighted,
}


# Builders
def build_model(model_cfg, data):
    if model_cfg.backend == "torch":
        return build_torch_model(model_cfg, data)

    if model_cfg.backend == "numpyro":
        return build_numpyro_model(model_cfg)

    raise ValueError(f"Unknown backend: {model_cfg.backend}")


def build_torch_model(model_cfg, data):
    if model_cfg.name not in TORCH_MODELS:
        raise KeyError(
            f"Unknown torch model '{model_cfg.name}'. "
            f"Available: {list(TORCH_MODELS.keys())}"
        )

    model_cls = TORCH_MODELS[model_cfg.name]
    X = data["X_train"]

    if model_cfg.params.num_quantiles is not None:
        quantiles = np.linspace(0, 1, model_cfg.params.num_quantiles)
    else:
        quantiles = np.linspace(0, 1, 51)

    return model_cls(
        num_features=X.shape[-1],
        num_parks=data["X_train"].shape[-2],
        quantiles=quantiles,
        target=model_cfg.params.target,
    )


def build_numpyro_model(model_cfg):
    if model_cfg.name not in NUMPYRO_MODELS:
        raise KeyError(
            f"Unknown numpyro model '{model_cfg.name}'. "
            f"Available: {list(NUMPYRO_MODELS.keys())}"
        )

    return NUMPYRO_MODELS[model_cfg.name]
