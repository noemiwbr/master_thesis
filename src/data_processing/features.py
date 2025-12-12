import pandas as pd
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler


# Use sin (North-South) and cos (East-West) instead of degree
def sin_cos_transform(df, columns=None, prefix=None, n_members=None):
    # Determine which columns to process (works for forecast and nowcast)
    if columns is not None:
        cols = columns
        new_cols_x = [f"{col}_x" for col in cols]
        new_cols_y = [f"{col}_y" for col in cols]
    elif prefix is not None and n_members is not None:
        cols = [f"{prefix}_member_{i}" for i in range(n_members)]
        new_cols_x = [f"{prefix}_x_member_{i}" for i in range(n_members)]
        new_cols_y = [f"{prefix}_y_member_{i}" for i in range(n_members)]
    else:
        raise ValueError("Either columns or (prefix + n_members) must be provided")

    angles_deg = df[cols].to_numpy()
    angles_rad = jnp.deg2rad(angles_deg)

    x = jnp.cos(angles_rad)
    y = jnp.sin(angles_rad)

    sin_cos_transformed = pd.DataFrame(
        jnp.concatenate([x, y], axis=1), columns=new_cols_x + new_cols_y, index=df.index
    )
    return pd.concat([df, sin_cos_transformed], axis=1).drop(cols, axis=1)


# Scaling of input features
def scale_features(train_df, test_df, scale_cols):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[scale_cols])
    # Transform test data using same scaler
    test_scaled = scaler.transform(test_df[scale_cols])
    train_scaled_df = pd.DataFrame(
        train_scaled, columns=scale_cols, index=train_df.index
    )
    test_scaled_df = pd.DataFrame(test_scaled, columns=scale_cols, index=test_df.index)

    # Keep non-scaled columns (e.g. wind dir - already transformed)
    missing_columns = [col for col in train_df.columns if col not in scale_cols]
    train_scaled_df[missing_columns] = train_df[missing_columns]
    test_scaled_df[missing_columns] = test_df[missing_columns]
    return train_scaled_df, test_scaled_df


# Fit scaler on training data and return parameters for later use.
def get_scaler_params(train_df, scale_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])
    return {
        "mean": dict(zip(scale_cols, scaler.mean_)),
        "scale": dict(zip(scale_cols, scaler.scale_)),
        "columns": scale_cols,
    }


# Apply pre-computed scaler parameters to new data.
def apply_scaler_params(df, scaler_params):
    scale_cols = scaler_params["columns"]
    mean = scaler_params["mean"]
    scale = scaler_params["scale"]

    scaled_df = df.copy()
    for col in scale_cols:
        scaled_df[col] = (df[col] - mean[col]) / scale[col]

    return scaled_df


# MultiIndex (timestamp, wind park) and feature columns -> np array with shape (timestamps, wind parks, features)
def get_input_shaped(X_df, feature_cols):
    # TODO aren't feature cols just the columns of the X_df??
    X_df = X_df.sort_index(level=[0, 1])

    timestamps = X_df.index.get_level_values(0).unique()
    windparks = X_df.index.get_level_values(1).unique()

    num_windparks = len(windparks)
    num_features = len(feature_cols)

    # Sanity check: full grid present
    expected_len = len(timestamps) * num_windparks
    assert len(X_df) == expected_len

    X_np = X_df[feature_cols].to_numpy(dtype=jnp.float32)
    X_np = X_np.reshape(len(timestamps), num_windparks, num_features)
    return X_np
