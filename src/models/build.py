import sys

sys.path.append("../src")

from models.model import (
    WindPowerQuantile,
    WindPowerQuantileEnsembles,
    model_mw_gamma,
    model_cf_beta,
)
import numpy as np

# Registries
TORCH_MODELS = {
    "WindPowerQuantile": WindPowerQuantile,
    "WindPowerQuantileEnsembles": WindPowerQuantileEnsembles,
}

NUMPYRO_MODELS = {
    "model_mw_gamma": model_mw_gamma,
    "model_cf_beta": model_cf_beta,
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

    if model_cfg.name in {
        "WindPowerQuantile",
        "WindPowerQuantileEnsembles",
    }:
        if model_cfg.params.quantiles is None:
            quantiles = np.linspace(0, 1, 51)
        else:
            quantiles = tuple(model_cfg.params.quantiles)

        return model_cls(
            num_features=X.shape[-1],
            quantiles=quantiles,
            target=model_cfg.params.target,
        )

    return model_cls(**model_cfg.params)


def build_numpyro_model(model_cfg):
    if model_cfg.name not in NUMPYRO_MODELS:
        raise KeyError(
            f"Unknown numpyro model '{model_cfg.name}'. "
            f"Available: {list(NUMPYRO_MODELS.keys())}"
        )

    return NUMPYRO_MODELS[model_cfg.name]
