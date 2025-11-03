import os
import pickle
from datetime import datetime
from glob import glob
import time
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.optim import Adam


def save_checkpoint(samples, model_name="bnn", args=None, rng_key=None,
                    dir_path="../checkpoints", target="gen"):

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


def load_checkpoint(model_name=None, file_name=None, dir_path="../checkpoints"):

    if file_name is not None:
        path_to_ckp = os.path.join(dir_path, f"{file_name}.pkl")
        print(f"Loading specified checkpoint: {path_to_ckp}")

    else: 
        pattern = os.path.join(dir_path, f"{model_name}_*.pkl")
        files = glob(pattern)

        if not files:
            raise FileNotFoundError(f"No checkpoint files found for model '{model_name}' in '{dir_path}'")

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
        num_warmup=args['num_warmup'],
        num_samples=args['num_samples'],
        num_chains=args['num_chains'],
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



def run_svi(model, guide, rng_key, X_train, Y_train, optimizer=Adam, lr=1e-3, num_steps=5000):
    optimizer = optimizer(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=8))

    svi_result = svi.run(rng_key, num_steps, X_train, Y_train, progress_bar=False)

    losses = svi_result.losses
    print(f"Final ELBO loss: {losses[-1]:.3f}")
    return svi_result
