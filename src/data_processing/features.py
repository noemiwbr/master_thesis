import pandas as pd
import jax.numpy as jnp
import re


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


def scale_features_nowcast(train_df, test_df, scale_cols, eps=1e-6):

    shifted_cols = [c for c in scale_cols if ("+" in c or "-" in c)]
    base_scale_cols = [c for c in scale_cols if c not in shifted_cols]

    # Fit stats on base cols only
    mean_train = train_df[base_scale_cols].mean()
    std_train = train_df[base_scale_cols].std().clip(lower=eps)

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Scale base cols normally
    train_scaled[base_scale_cols] = (train_df[base_scale_cols] - mean_train) / std_train
    test_scaled[base_scale_cols] = (test_df[base_scale_cols] - mean_train) / std_train

    # Scale shifted cols using their base column stats
    base_re = re.compile(r"^(.+?)([+-]\d+)$")
    for col in shifted_cols:
        base_col = base_re.match(col).group(1)

        mu = mean_train[base_col]
        sigma = std_train[base_col]
        train_scaled[col] = (train_df[col] - mu) / sigma
        test_scaled[col] = (test_df[col] - mu) / sigma

    # Return params for saving
    scaler_params = {
        "mean": mean_train.to_dict(),
        "std": std_train.to_dict(),
        "columns": base_scale_cols,
        "shifted_columns": shifted_cols,
    }

    return train_scaled, test_scaled, scaler_params


def apply_scale_features_nowcast(test_df, params):

    shifted_cols = params["shifted_columns"]
    base_scale_cols = params["columns"]

    mean_train = params["mean"]
    std_train = params["std"]

    test_scaled = test_df.copy()

    # Scale base cols normally
    test_scaled[base_scale_cols] = (test_df[base_scale_cols] - mean_train) / std_train

    # Scale shifted cols using their base column stats
    base_re = re.compile(r"^(.+?)([+-]\d+)$")
    for col in shifted_cols:
        base_col = base_re.match(col).group(1)

        mu = mean_train[base_col]
        sigma = std_train[base_col]
        test_scaled[col] = (test_df[col] - mu) / sigma

    return test_scaled


def scale_forecast_with_nowcast_scalers(
    forecast_df,
    nowcast_scaler_params,
    eps=1e-6,
):
    """
    Scales forecast ensemble-member columns using nowcast mean/std.
    - x_member_i uses mean/std from x
    - x+1_member_i (or x-1_member_i etc) uses mean/std from base x
    """
    mean = nowcast_scaler_params["mean"]
    std = nowcast_scaler_params["std"]

    # safety: avoid divide-by-zero
    std = {k: max(float(v), eps) for k, v in std.items()}

    scaled = forecast_df.copy()

    _MEMBER_RE = re.compile(r"^(?P<name>.+)_member_(?P<i>\d+)$")
    _SHIFTED_MEMBER_RE = re.compile(
        r"^(?P<base>.+?)(?P<shift>[+-]\d+)_member_(?P<i>\d+)$"
    )

    for col in forecast_df.columns:

        m_shift = _SHIFTED_MEMBER_RE.match(col)
        m_mem = _MEMBER_RE.match(col)

        # 1. shifted member columns: e.g. wind_speed_10m+1_member_i
        if m_shift:
            base = m_shift.group("base")  # e.g. wind_speed_10m
            scaled[col] = (forecast_df[col] - mean[base]) / std[base]

        # 2. normal member columns: e.g. air_temperature_2m_member_i
        elif m_mem:
            base = m_mem.group("name")
            if base in mean and base in std:
                scaled[col] = (forecast_df[col] - mean[base]) / std[base]
            # else: leave as-is (not a scaled variable). e.g. wind_direction_10m_x

        # 3. non-member base cols, e.g. hour_x (if we scale it, right now not in use)
        elif col in mean and col in std:
            scaled[col] = (forecast_df[col] - mean[col]) / std[col]

        # 4. non-scaled (non-member) columns, e.g. hour_x, windpark_id,...
        # else: leave as-is

    return scaled


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
