import os
import pickle
from datetime import datetime
from glob import glob
import time
from numpyro import handlers
from numpyro.infer import MCMC, NUTS


def save_checkpoint(samples, model_name="bnn", args=None, rng_key=None,
                    dir_path="../../checkpoints", target="gen"):

    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_{timestamp}_{target}.pkl"
    path = os.path.join(dir_path, filename)

    # Bundle all relevant information into one dictionary
    state = {
        "model_name": model_name,
        "timestamp": timestamp,
        "samples": samples,
        "args": args if args is not None else None,
        "rng_key": rng_key,
    }

    with open(path, "wb") as f:
        pickle.dump(state, f)

    print(f"Saved checkpoint for model '{model_name}': {path}")
    return path


# TODO explicit name
def load_latest_checkpoint(model_name, dir_path="../checkpoints"):

    pattern = os.path.join(dir_path, f"{model_name}_*.pkl")
    files = glob(pattern)

    if not files:
        raise FileNotFoundError(f"No checkpoint files found for model '{model_name}' in '{dir_path}'")

    # Sort by modification time and pick the latest
    files.sort(key=os.path.getmtime)
    latest_path = files[-1]

    print(f"Loading latest checkpoint for '{model_name}': {latest_path}")

    with open(latest_path, "rb") as f:
        state = pickle.load(f)

    return state



# Inference helper for basic BNN
def run_inference_bnn(model, args, rng_key, X, Y, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args['num_warmup'],
        num_samples=args['num_samples'],
        num_chains=args['num_chains'],
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(rng_key, X=X, Y=Y, D_H=D_H)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()



# Posterior predictive for basic BNN
def predict_bnn(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return trace["Y"]["value"]