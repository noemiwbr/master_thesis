import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from data_processing.preprocess import (
    filter_bid_zone,
    align_windpower_and_weather,
    merge_metadata,
    multiindex,
    capacity_factor,
    split_train_test_meta,
)
from data_processing.features import (
    winddir_transform,
    scale_features,
    get_input_shaped,
    get_scaler_params,
    apply_scaler_params,
)


def load_weather_forecast(path: str):
    df = pd.read_parquet(path)
    df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert("CET")
    df["time_ref"] = df["time_ref"].dt.tz_localize("UTC").dt.tz_convert("CET")
    return df


def load_weather_nowcast(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize("UTC").tz_convert("CET")


def load_windpower(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize("UTC").tz_convert("CET")


def load_windparks(path: str):
    return pd.read_csv(path)


def load_and_preprocess_data(
    bid_zone="ELSPOT NO1",
    split_time=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
    data_path="../data",
):
    # Load data
    weather_nowcast = load_weather_nowcast(f"{data_path}/met_nowcast.parquet")
    windpower = load_windpower(f"{data_path}/wind_power_per_bidzone.parquet")
    windparks = load_windparks(f"{data_path}/windparks_bidzone.csv")

    # Apply wind direction transformation
    # _winddir_transformed_forecast = winddir_transform(weather_forecast, prefix='wd10m_', n_members=15)
    _winddir_transformed_nowcast = winddir_transform(
        weather_nowcast, columns=["wind_direction_10m"]
    )
    weather_nowcast = pd.concat([weather_nowcast, _winddir_transformed_nowcast], axis=1)

    # Preprocessing pipeline
    windparks_in_bid_zone = filter_bid_zone(windparks, bid_zone)
    _windpower_in_bid_zone = windpower[bid_zone].dropna()

    # Align windpower and weather, only keep timestamps which are present in both datasets
    _weather_nowcast_in_bid_zone, _windpower_in_bid_zone = align_windpower_and_weather(
        _windpower_in_bid_zone, weather_nowcast
    )

    _weather_nowcast_in_bid_zone = merge_metadata(
        _weather_nowcast_in_bid_zone, windparks_in_bid_zone
    )

    # MultiIndex (time, windpark), operating_power_max is NaN for windparks not yet operational
    weather_nowcast_in_bid_zone = multiindex(
        _weather_nowcast_in_bid_zone, windparks_in_bid_zone
    )

    # Capacity factor
    windpower_in_bid_zone = capacity_factor(bid_zone, _windpower_in_bid_zone)

    assert (_windpower_in_bid_zone.index == windpower_in_bid_zone.index).all()

    # Only use timestamps where all wind parks are operational
    # TODO find a better way how to deal with that
    start_time = windparks_in_bid_zone["prod_start_new"].max()
    windpower_in_bid_zone = windpower_in_bid_zone.loc[start_time:]
    weather_nowcast_in_bid_zone = weather_nowcast_in_bid_zone.loc[start_time:]

    # Split into train and test data
    feature_cols = [
        "air_temperature_2m",
        "air_pressure_at_sea_level",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m_x",
        "wind_direction_10m_y",
    ]
    site_meta_cols = ["operating_power_max", "windpark_id"]
    scale_cols = [c for c in feature_cols if not c.startswith("wind_direction")]

    df_train_X, df_train_Y, df_test_X, df_test_Y, sites_meta_unscaled = (
        split_train_test_meta(
            weather_nowcast_in_bid_zone,
            windpower_in_bid_zone,
            feature_cols,
            site_meta_cols,
            split_time,
        )
    )

    # Scaling
    df_train_X_scaled, df_test_X_scaled = scale_features(
        df_train_X, df_test_X, scale_cols
    )

    # Get scaler parameters for later use with evaluation data
    scaler_params = get_scaler_params(df_train_X, scale_cols)

    mean_sites = sites_meta_unscaled.mean()
    std_sites = sites_meta_unscaled.std()
    sites_meta = ((sites_meta_unscaled - mean_sites) / std_sites)[:, None]

    # Package site metadata scaling params
    site_scaler_params = {"mean": mean_sites, "std": std_sites}

    # Convert to numpy arrays
    X_train = get_input_shaped(df_train_X_scaled, feature_cols)
    X_test = get_input_shaped(df_test_X_scaled, feature_cols)

    Y_train_cf = df_train_Y["capacity_factor"].to_numpy(jnp.float32)
    Y_train_aval = df_train_Y["available_capacity"].to_numpy(jnp.float32)
    Y_train_gen = df_train_Y["generation"].to_numpy(jnp.float32)

    Y_test_cf = df_test_Y["capacity_factor"].to_numpy(jnp.float32)
    Y_test_aval = df_test_Y["available_capacity"].to_numpy(jnp.float32)
    Y_test_gen = df_test_Y["generation"].to_numpy(jnp.float32)

    assert (
        weather_nowcast_in_bid_zone.index.get_level_values("time").nunique()
        == windpower_in_bid_zone.index.nunique()
    )
    assert np.isclose(
        windpower_in_bid_zone["available_capacity"]
        * windpower_in_bid_zone["capacity_factor"],
        windpower_in_bid_zone["generation"],
    ).all()
    assert df_train_X.index.get_level_values("time").unique().equals(df_train_Y.index)
    assert df_test_X.index.get_level_values("time").unique().equals(df_test_Y.index)

    print("Preprocessing done. X_train shape:", X_train.shape)
    print("Preprocessing done. X_test shape:", X_test.shape)

    return dict(
        X_train=X_train,
        X_test=X_test,
        Y_train_gen=Y_train_gen,
        Y_test_gen=Y_test_gen,
        Y_train_cf=Y_train_cf,
        Y_test_cf=Y_test_cf,
        Y_train_aval=Y_train_aval,
        Y_test_aval=Y_test_aval,
        sites_meta=sites_meta,
        feature_cols=feature_cols,
        scaler_params=scaler_params,
        site_scaler_params=site_scaler_params,
    )


# Load and preprocess evaluation data for a specific date, use scalar from training set
def load_evaluation_data(
    date,
    scaler_params,
    site_scaler_params,
    bid_zone="ELSPOT NO1",
    data_path="../data",
):

    # Parse and format date
    if isinstance(date, str):
        # Try different date formats
        try:
            # Try YYYY-MM-DD format
            date_parsed = pd.to_datetime(date, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date format not recognized: {date}. "
                "Please use 'YYYY-MM-DD' format."
            )
    else:
        raise TypeError(f"Date must be string, got {type(date)}")

    # Format date as YYYYMMDD for file names
    date_str = date_parsed.strftime("%Y%m%d")

    # Construct file paths
    path_to_weather = os.path.join(
        data_path, "evaluation", "met_nowcast_daily", f"met_nowcast_{date_str}.parquet"
    )
    path_to_windpower = os.path.join(
        data_path,
        "evaluation",
        "wind_power",
        f"wind_power_per_bidzone_{date_str}.parquet",
    )

    # Check if files exist
    if not os.path.exists(path_to_weather):
        raise FileNotFoundError(
            f"Weather nowcast file not found: {path_to_weather}\n"
            f"Expected file for date {date_parsed.strftime('%Y-%m-%d')}"
        )

    if not os.path.exists(path_to_windpower):
        raise FileNotFoundError(
            f"Wind power file not found: {path_to_windpower}\n"
            f"Expected file for date {date_parsed.strftime('%Y-%m-%d')}"
        )

    print(
        f"Loading evaluation data for {date_parsed.strftime('%Y-%m-%d')} in zone {bid_zone}"
    )

    # Load data
    weather_nowcast = load_weather_nowcast(path_to_weather)
    windpower = load_windpower(path_to_windpower)

    # Load windparks metadata (not date-specific)
    windparks = load_windparks(f"{data_path}/windparks_bidzone.csv")

    # Apply wind direction transformation
    _winddir_transformed_nowcast = winddir_transform(
        weather_nowcast, columns=["wind_direction_10m"]
    )
    weather_nowcast = pd.concat([weather_nowcast, _winddir_transformed_nowcast], axis=1)

    # Preprocessing pipeline
    windparks_in_bid_zone = filter_bid_zone(windparks, bid_zone)
    _windpower_in_bid_zone = windpower[bid_zone]

    # Check for NaN values before alignment
    if _windpower_in_bid_zone.isna().any():
        nan_count = _windpower_in_bid_zone.isna().sum()
        raise ValueError(
            f"Found {nan_count} NaN value(s) in wind power data for {bid_zone}. "
            "Evaluation data should not contain missing values."
        )
    if weather_nowcast.isna().any().any():
        nan_cols = weather_nowcast.columns[weather_nowcast.isna().any()].tolist()
        nan_counts = weather_nowcast.isna().sum()[weather_nowcast.isna().any()]
        raise ValueError(
            f"Found NaN values in nowcast data columns: {dict(zip(nan_cols, nan_counts))}. "
            "Evaluation data should not contain missing values."
        )

    # Align windpower and weather, only keep timestamps which are present in both datasets
    _weather_nowcast_in_bid_zone, _windpower_in_bid_zone = align_windpower_and_weather(
        _windpower_in_bid_zone, weather_nowcast
    )

    _weather_nowcast_in_bid_zone = merge_metadata(
        _weather_nowcast_in_bid_zone, windparks_in_bid_zone
    )

    # MultiIndex (time, windpark), operating_power_max is NaN for windparks not yet operational
    weather_nowcast_in_bid_zone = multiindex(
        _weather_nowcast_in_bid_zone, windparks_in_bid_zone
    )

    # Capacity factor
    windpower_in_bid_zone = capacity_factor(bid_zone, _windpower_in_bid_zone)

    assert (_windpower_in_bid_zone.index == windpower_in_bid_zone.index).all()

    # Assert timestamps align between weather and wind power
    weather_timestamps = weather_nowcast_in_bid_zone.index.get_level_values(
        "time"
    ).unique()
    windpower_timestamps = windpower_in_bid_zone.index
    assert weather_timestamps.equals(
        windpower_timestamps
    ), f"Timestamp mismatch: weather has {len(weather_timestamps)} unique times, wind power has {len(windpower_timestamps)}"

    feature_cols = [
        "air_temperature_2m",
        "air_pressure_at_sea_level",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m_x",
        "wind_direction_10m_y",
    ]
    site_meta_cols = ["operating_power_max", "windpark_id"]

    # Extract site metadata
    sites_meta_unscaled = (
        weather_nowcast_in_bid_zone[site_meta_cols]
        .drop_duplicates()
        .dropna()
        .sort_values("windpark_id")["operating_power_max"]
        .to_numpy()
    )

    # Now select only feature columns
    weather_nowcast_in_bid_zone = weather_nowcast_in_bid_zone[feature_cols]

    # Apply scaling using training scaler parameters
    weather_nowcast_scaled = apply_scaler_params(
        weather_nowcast_in_bid_zone, scaler_params
    )

    # Apply site metadata scaling using training parameters
    sites_meta = (
        (sites_meta_unscaled - site_scaler_params["mean"]) / site_scaler_params["std"]
    )[:, None]

    # Convert to numpy arrays
    X_eval = get_input_shaped(weather_nowcast_scaled, feature_cols)

    Y_eval_cf = windpower_in_bid_zone["capacity_factor"].to_numpy(jnp.float32)
    Y_eval_aval = windpower_in_bid_zone["available_capacity"].to_numpy(jnp.float32)
    Y_eval_gen = windpower_in_bid_zone["generation"].to_numpy(jnp.float32)

    print(f"Evaluation data shape: X={X_eval.shape}, Y_gen={Y_eval_gen.shape}")

    return dict(
        X_eval=X_eval,
        Y_eval_gen=Y_eval_gen,
        Y_eval_cf=Y_eval_cf,
        Y_eval_aval=Y_eval_aval,
        sites_meta=sites_meta,
        feature_cols=feature_cols,
        index=weather_timestamps,
    )
