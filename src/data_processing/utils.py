import os
import pickle
from datetime import datetime
from glob import glob
import time
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam


def save_checkpoint(
    samples,
    model_name="bnn",
    args=None,
    rng_key=None,
    rng_key_predict=None,
    dir_path="../checkpoints",
    target="gen",
    bid_zone="ELSPOT_NO1",
):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_{timestamp}_{target}_{bid_zone}.pkl"
    path = os.path.join(dir_path, filename)

    # Bundle all relevant information into one dictionary
    state = {
        "model_name": model_name,
        "timestamp": timestamp,
        "samples": samples,
        "args": args,
        "rng_key": rng_key,
        "rng_key_predict": rng_key_predict,
        "bid_zone": bid_zone,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved checkpoint for model '{model_name}': {path}")
    return path


# Save scaler parameters for a specific bid zone and split time
def save_scaler_params(
    scaler_params,
    site_scaler_params,
    bid_zone="ELSPOT_NO1",
    split_time=None,
    dir_path="../checkpoints",
):

    os.makedirs(dir_path, exist_ok=True)

    # Clean bid zone name for filename
    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"scaler_params_{bid_zone_clean}.pkl"
    path = os.path.join(dir_path, filename)

    state = {
        "bid_zone": bid_zone,
        "split_time": str(split_time) if split_time is not None else None,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scaler_params": scaler_params,
        "site_scaler_params": site_scaler_params,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved scaler parameters for {bid_zone}: {path}")
    return path


# Load scaler parameters for a specific bid zone
def load_scaler_params(bid_zone="ELSPOT_NO1", dir_path="../checkpoints"):

    bid_zone_clean = bid_zone.replace(" ", "_")
    filename = f"scaler_params_{bid_zone_clean}.pkl"
    path = os.path.join(dir_path, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No scaler parameters found for {bid_zone} at {path}\n"
            f"Run load_and_preprocess_data() and save_scaler_params() first."
        )

    with open(path, "rb") as f:
        state = pickle.load(f)

    print(f"Loaded scaler parameters for {state['bid_zone']}")
    print(f"  Saved at: {state['saved_at']}")
    if state["split_time"]:
        print(f"  Split time: {state['split_time']}")

    return {
        "scaler_params": state["scaler_params"],
        "site_scaler_params": state["site_scaler_params"],
    }


def load_checkpoint(model_name=None, file_name=None, dir_path="../checkpoints"):

    if file_name is not None:
        path_to_ckp = os.path.join(dir_path, f"{file_name}.pkl")
        print(f"Loading specified checkpoint: {path_to_ckp}")

    else:
        pattern = os.path.join(dir_path, f"{model_name}_*.pkl")
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No checkpoint files found for model '{model_name}' in '{dir_path}'"
            )

        # Sort by modification time and pick the latest
        files.sort(key=os.path.getmtime)
        path_to_ckp = files[-1]

        print(f"Loading latest checkpoint: {path_to_ckp}")

    with open(path_to_ckp, "rb") as f:
        state = pickle.load(f)

    return state


# TODO more generic, should be possible to have quanitles as argument
# Inference helper for basic BNN
def run_inference_bnn(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(rng_key, X=X, Y=Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# Posterior predictive for basic BNN
def predict_bnn(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(model).get_trace(X=X, Y=None)
    return trace["Y"]["value"]


def run_svi(
    model, guide, rng_key, X_train, Y_train, optimizer=Adam, lr=1e-3, num_steps=5000
):
    optimizer = optimizer(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=8))

    svi_result = svi.run(rng_key, num_steps, X_train, Y_train, progress_bar=False)

    losses = svi_result.losses
    print(f"Final ELBO loss: {losses[-1]:.3f}")
    return svi_result
