import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import re
from data_processing.preprocess import (
    filter_bid_zone,
    align_windpower_and_weather,
    merge_metadata,
    multiindex,
    capacity_factor,
    split_train_test_meta,
)
from data_processing.features import (
    sin_cos_transform,
    scale_features,
    get_input_shaped,
    get_scaler_params,
    apply_scaler_params,
)


def load_weather_forecast(path: str):
    df = pd.read_csv(path, parse_dates=["time", "time_ref"], dtype={"sid": str})
    df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert("CET")
    df["time_ref"] = df["time_ref"].dt.tz_localize("UTC").dt.tz_convert("CET")
    return df


def load_weather_nowcast(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize("UTC").tz_convert("CET")


def load_weather_forecast_parquet(path: str):
    df = pd.read_parquet(path)
    # Convert time column to CET (forecast has time as column, not index)
    if "time" in df.columns:
        df["time"] = (
            pd.to_datetime(df["time"]).dt.tz_localize("UTC").dt.tz_convert("CET")
        )
    if "time_ref" in df.columns:
        df["time_ref"] = (
            pd.to_datetime(df["time_ref"]).dt.tz_localize("UTC").dt.tz_convert("CET")
        )
    return df


def load_windpower(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize("UTC").tz_convert("CET")


def load_windparks(path: str):
    return pd.read_csv(path)


def get_feature_columns():
    """Define standard feature and metadata columns"""

    feature_cols = [
        "air_temperature_2m",
        "air_pressure_at_sea_level",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_speed_10m-1",
        "wind_speed_10m+1",
        "wind_direction_10m_x",
        "wind_direction_10m_y",
        "hour_x",
        "hour_y",
    ]
    site_meta_cols = ["operating_power_max", "windpark_id"]
    scale_cols = [c for c in feature_cols if not c.startswith("wind_direction")]

    return feature_cols, site_meta_cols, scale_cols


def prepare_bid_zone_data(weather_df, windpower_df, windparks_df, bid_zone):
    """Filter and align data for a specific bidding zone, calculate capacity factor"""
    # Filter by bid zone
    windparks_in_bid_zone = filter_bid_zone(windparks_df, bid_zone)
    windpower_in_bid_zone = windpower_df[bid_zone].dropna()

    # Align weather and wind power on timestamps
    weather_aligned, windpower_aligned = align_windpower_and_weather(
        windpower_in_bid_zone, weather_df
    )

    # Merge wind park metadata into weather
    weather_with_meta = merge_metadata(weather_aligned, windparks_in_bid_zone)

    # Create MultiIndex (time, windpark)
    weather_multiindex = multiindex(weather_with_meta, windparks_in_bid_zone)

    # Calculate capacity factor
    windpower_with_cf = capacity_factor(bid_zone, windpower_aligned)

    return weather_multiindex, windpower_with_cf


def filter_operational_period(weather_df, windpower_df, windparks_df):
    """Filter data to only include period where all wind parks are operational"""

    start_time = windparks_df["prod_start_new"].max()
    weather_filtered = weather_df.loc[start_time:]
    windpower_filtered = windpower_df.loc[start_time:]

    return weather_filtered, windpower_filtered


def scale_site_metadata(sites_meta_unscaled, scaler_params=None):
    """Scale site metadata, fits new scaler if params not provided"""
    if scaler_params is None:
        # Fit new scaler (training mode)
        mean_sites = sites_meta_unscaled.mean()
        std_sites = sites_meta_unscaled.std()
        scaler_params = {"mean": mean_sites, "std": std_sites}
    else:
        # Use existing scaler (evaluation mode)
        mean_sites = scaler_params["mean"]
        std_sites = scaler_params["std"]

    sites_meta = ((sites_meta_unscaled - mean_sites) / std_sites)[:, None]
    return sites_meta, scaler_params


# TODO only used once
def scale_and_extract_metadata(df_train_X, df_test_X, sites_meta_unscaled, scale_cols):
    """
    Scale features and site metadata, return scaler parameters.

    Args:
        df_train_X: Training features
        df_test_X: Test features
        sites_meta_unscaled: Unscaled site metadata
        scale_cols: Columns to scale (excludes wind direction)

    Returns:
        Dict with scaled data and scaler parameters
    """
    # Scale features
    df_train_X_scaled, df_test_X_scaled = scale_features(
        df_train_X, df_test_X, scale_cols
    )

    # Get scaler parameters for later use
    scaler_params = get_scaler_params(df_train_X, scale_cols)

    # Scale site metadata
    sites_meta, site_scaler_params = scale_site_metadata(sites_meta_unscaled)

    return {
        "df_train_X_scaled": df_train_X_scaled,
        "df_test_X_scaled": df_test_X_scaled,
        "sites_meta": sites_meta,
        "scaler_params": scaler_params,
        "site_scaler_params": site_scaler_params,
    }


def convert_to_arrays(df_train_X, df_test_X, df_train_Y, df_test_Y, feature_cols):
    """Convert DataFrames to numpy/JAX arrays with correct shapes"""

    X_train = get_input_shaped(df_train_X, feature_cols)
    X_test = get_input_shaped(df_test_X, feature_cols)

    Y_train = convert_targets_to_arrays(df_train_Y, "Y_train_")
    Y_test = convert_targets_to_arrays(df_test_Y, "Y_test_")

    return (
        {
            "X_train": X_train,
            "X_test": X_test,
        }
        | Y_train
        | Y_test
    )


def convert_targets_to_arrays(windpower_df, prefix=""):
    """Convert wind power targets to numpy/JAX arrays"""
    return {
        f"{prefix}cf": windpower_df["capacity_factor"].to_numpy(jnp.float32),
        f"{prefix}aval": windpower_df["available_capacity"].to_numpy(jnp.float32),
        f"{prefix}gen": windpower_df["generation"].to_numpy(jnp.float32),
    }


def validate_preprocessing(
    weather_df, windpower_df, df_train_X, df_train_Y, df_test_X, df_test_Y
):
    """Run assertions to validate data integrity after preprocessing"""

    # Check timestamps and capacity factor alignment
    weather_timestamps = weather_df.index.get_level_values("time").unique()
    windpower_timestamps = windpower_df.index

    assert weather_timestamps.equals(windpower_timestamps), (
        f"Timestamp mismatch: weather has {len(weather_timestamps)} unique times, "
        f"wind power has {len(windpower_timestamps)}"
    )

    # Check capacity factor calculation
    assert np.isclose(
        windpower_df["available_capacity"] * windpower_df["capacity_factor"],
        windpower_df["generation"],
    ).all(), "Capacity factor calculation error"

    # Check train/test split alignment
    assert (
        df_train_X.index.get_level_values("time").unique().equals(df_train_Y.index)
    ), "Training data timestamps don't align"

    assert (
        df_test_X.index.get_level_values("time").unique().equals(df_test_Y.index)
    ), "Test data timestamps don't align"


def load_and_preprocess_data(
    bid_zone="ELSPOT NO1",
    split_time=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
    data_path="../data",
):
    """
    Load and preprocess training data for wind power forecasting.

    This is the main entry point for data preprocessing. The function follows
    a pipeline:
    1. Load raw data
    2. Add time related features (hour of day, shifted wind speed)
    3. Transform features (wind direction)
    4. Filter and align data for bid zone
    5. Filter operational period
    6. Split train/test
    7. Scale features
    8. Convert to arrays

    Args:
        bid_zone: Bidding zone name (e.g., "ELSPOT NO1")
        split_time: Timestamp to split train/test
        data_path: Path to data directory

    Returns:
        Dict with training and test data, scaler parameters
    """
    print(f"Loading data for {bid_zone}...")

    # 1. Load raw data
    weather_nowcast = load_weather_nowcast(f"{data_path}/met_nowcast.parquet")
    windpower = load_windpower(f"{data_path}/wind_power_per_bidzone.parquet")
    windparks = load_windparks(f"{data_path}/windparks_bidzone.csv")

    # 2. Add features (lagged wind speed, hour of the day)
    weather_nowcast = add_shifted_features(weather_nowcast)

    # 3. Transform wind direction and hour to sin/cos components
    weather_nowcast = sin_cos_transform(
        weather_nowcast, columns=["wind_direction_10m", "hour"]
    )

    # 4. Filter and align data for specific bid zone, calculate capacity factor
    weather_multiindex, windpower_with_cf = prepare_bid_zone_data(
        weather_nowcast, windpower, windparks, bid_zone
    )

    # 5. Filter period where all wind parks are operational
    windparks_in_bid_zone = filter_bid_zone(windparks, bid_zone)
    weather_filtered, windpower_filtered = filter_operational_period(
        weather_multiindex, windpower_with_cf, windparks_in_bid_zone
    )

    # 6. Split into train and test sets
    feature_cols, site_meta_cols, scale_cols = get_feature_columns()

    df_train_X, df_train_Y, df_test_X, df_test_Y, sites_meta_unscaled = (
        split_train_test_meta(
            weather_filtered,
            windpower_filtered,
            feature_cols,
            site_meta_cols,
            split_time,
        )
    )

    # 7. Scale features and metadata
    df_train_X_scaled, df_test_X_scaled = scale_features(
        df_train_X, df_test_X, scale_cols
    )

    # Get scaler parameters for later use
    scaler_params = get_scaler_params(df_train_X, scale_cols)

    # Scale site metadata
    sites_meta, site_scaler_params = scale_site_metadata(sites_meta_unscaled)

    scaling_result = {
        "df_train_X_scaled": df_train_X_scaled,
        "df_test_X_scaled": df_test_X_scaled,
        "sites_meta": sites_meta,
        "scaler_params": scaler_params,
        "site_scaler_params": site_scaler_params,
    }

    # 8. Convert to numpy/JAX arrays
    arrays = convert_to_arrays(
        scaling_result["df_train_X_scaled"],
        scaling_result["df_test_X_scaled"],
        df_train_Y,
        df_test_Y,
        feature_cols,
    )

    # Validate data integrity
    validate_preprocessing(
        weather_filtered,
        windpower_filtered,
        df_train_X,
        df_train_Y,
        df_test_X,
        df_test_Y,
    )

    print(f"Preprocessing done. X_train shape: {arrays['X_train'].shape}")
    print(f"Preprocessing done. X_test shape: {arrays['X_test'].shape}")

    return {
        **arrays,
        "sites_meta": scaling_result["sites_meta"],
        "feature_cols": feature_cols,
        "scaler_params": scaling_result["scaler_params"],
        "site_scaler_params": scaling_result["site_scaler_params"],
    }


def load_and_preprocess_forecast_data(
    scaler_params=None,
    site_scaler_params=None,
    bid_zone="ELSPOT NO1",
    split_time=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
    ensemble_member=0,
    data_path="../data",
):
    """
    Load and preprocess forecast training data using existing scaler parameters

    Pipeline (same as for nowcast):
    1. Load raw forecast data (parquet)
    2. Rename columns and select ensemble member
    3. Transform features (wind direction)
    4. Calculate variance across ensemble members
    5. Add time related features (hour of day, shifted wind speed)
    6. Filter and align data for bid zone
    7. Filter operational period
    8. Split train/test
    9. Scaling (apply existing scaler parameters or calculate new ones)
    10. Convert to arrays
    """
    print(f"Loading forecast data for {bid_zone}, ensemble member {ensemble_member}...")

    # 1. Load raw data
    weather_forecast = load_weather_forecast_parquet(
        f"{data_path}/met_forecast.parquet"
    )
    windpower = load_windpower(f"{data_path}/wind_power_per_bidzone.parquet")
    windparks = load_windparks(f"{data_path}/windparks_bidzone.csv")

    feature_cols, site_meta_cols, scale_cols = get_feature_columns()

    # 2. Rename columns and select ensemble member
    weather_forecast_renamed = rename_met_forecast_columns(weather_forecast)

    # 3. Transform features (wind direction)
    # ----variance calculation----
    # TODO put variance calculation into function
    # Wind direction transformation of all ensemble members to get correct variance
    weather_forecast_transformed = sin_cos_transform(
        weather_forecast_renamed, prefix="wind_direction_10m", n_members=15
    )

    # 4. Calculate variance across ensemble members
    exclude = {
        "hour_x",
        "hour_y",
        "wind_speed_10m-1",
        "wind_speed_10m+1",
    }

    feature_member_cols = {
        feat: [f"{feat}_member_{i}" for i in range(15)]
        for feat in feature_cols
        if feat not in exclude
    }
    var_by_feature = {}
    for feat, cols in feature_member_cols.items():
        var_by_feature[feat] = (
            weather_forecast_transformed.groupby("lead_time_hours")[cols]
            .var()
            .mean(axis=1)
        )
    weather_vars = pd.DataFrame(var_by_feature)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    weather_vars_scaled = pd.DataFrame(
        scaler.fit_transform(weather_vars),
        index=weather_vars.index,
        columns=weather_vars.columns,
    )
    overall_variance = weather_vars_scaled.mean(axis=1)

    weather_forecast_transformed = weather_forecast_transformed.join(
        overall_variance.rename("variance"), on="lead_time_hours"
    )

    feature_cols += ["variance"]
    # ----variance calculation----

    weather_forecast_selected = select_ensemble_member(
        weather_forecast_transformed, member=ensemble_member
    )
    print(len(weather_forecast_selected.columns))

    # Filter to keep only 2-day ahead forecast (48h window starting at day+2 midnight)
    print(f"Before 2-day ahead filtering: {len(weather_forecast_selected)} rows")
    forecast_day = weather_forecast_selected["time"].dt.floor("D")
    ref_day_plus_2 = weather_forecast_selected["time_ref"].dt.floor("D") + pd.Timedelta(
        days=2
    )

    # Keep only rows where prediction day = reference day + 2
    weather_forecast_selected = weather_forecast_selected[
        forecast_day == ref_day_plus_2
    ]

    print(f"After 2-day ahead filtering: {len(weather_forecast_selected)} rows")
    print(
        f"Lead time range: {weather_forecast_selected['lead_time_hours'].min():.1f}h "
        f"to {weather_forecast_selected['lead_time_hours'].max():.1f}h"
    )

    weather_forecast_selected = weather_forecast_selected.set_index("time")

    # 5. Add time related features (hour of day, shifted wind speed)
    weather_forecast_selected = add_shifted_features(weather_forecast_selected)

    # Transform wind direction to sin/cos components
    weather_forecast_selected = sin_cos_transform(
        weather_forecast_selected, columns=["hour"]
    )

    # 6. Filter and align data for specific bid zone, calculate capacity factor
    weather_multiindex, windpower_with_cf = prepare_bid_zone_data(
        weather_forecast_selected, windpower, windparks, bid_zone
    )

    # 7. Filter period where all wind parks are operational
    windparks_in_bid_zone = filter_bid_zone(windparks, bid_zone)
    weather_filtered, windpower_filtered = filter_operational_period(
        weather_multiindex, windpower_with_cf, windparks_in_bid_zone
    )

    # 8. Split into train and test sets

    df_train_X, df_train_Y, df_test_X, df_test_Y, sites_meta_unscaled = (
        split_train_test_meta(
            weather_filtered,
            windpower_filtered,
            feature_cols,
            site_meta_cols,
            split_time,
        )
    )

    # 9. Scaling TODO: Do I need scalar values later?
    # Apply existing scaler parameters (from nowcast training)
    if scaler_params is None:
        print("No feature scaler parameter provided, computing new ones.")
        df_train_X_scaled, df_test_X_scaled = scale_features(
            df_train_X, df_test_X, scale_cols
        )

        # Compute new site metadata scaler
        sites_meta, site_scaler_params = scale_site_metadata(
            sites_meta_unscaled, scaler_params=None
        )

        # Save scaler params
        scaler_params = {
            "mean": df_train_X[scale_cols].mean(),
            "std": df_train_X[scale_cols].std(),
        }

    # Use provided scalar parameters
    else:
        print("Using provided scaler parameters.")
        df_train_X_scaled = apply_scaler_params(df_train_X, scaler_params)
        df_test_X_scaled = apply_scaler_params(df_test_X, scaler_params)

        # Use provided scaler for site metadata
        sites_meta, _ = scale_site_metadata(sites_meta_unscaled, site_scaler_params)

    # 10. Convert to numpy/JAX arrays
    arrays = convert_to_arrays(
        df_train_X_scaled,
        df_test_X_scaled,
        df_train_Y,
        df_test_Y,
        feature_cols,
    )

    # Validate data integrity
    validate_preprocessing(
        weather_filtered,
        windpower_filtered,
        df_train_X,
        df_train_Y,
        df_test_X,
        df_test_Y,
    )

    print(f"Preprocessing done. X_train shape: {arrays['X_train'].shape}")
    print(f"Preprocessing done. X_test shape: {arrays['X_test'].shape}")

    # Return combined results
    return {
        **arrays,
        "sites_meta": sites_meta,
        "feature_cols": feature_cols,
        "scaler_params": scaler_params,
        "site_scaler_params": site_scaler_params,
        "feature_cols": feature_cols,
    }


# Wind power shifted +/- 1 hour and hour as additional feature
def add_shifted_features(weather_forecast_selected):
    weather_forecast_selected["wind_speed_10m-1"] = weather_forecast_selected.groupby(
        "windpark"
    )["wind_speed_10m"].shift(-1)
    weather_forecast_selected["wind_speed_10m+1"] = weather_forecast_selected.groupby(
        "windpark"
    )["wind_speed_10m"].shift(+1)
    weather_forecast_selected = weather_forecast_selected.dropna(
        subset=["wind_speed_10m-1", "wind_speed_10m+1"]
    )
    weather_forecast_selected["hour"] = weather_forecast_selected.index.hour
    return weather_forecast_selected


def parse_evaluation_date(date):
    """Parse and validate evaluation date (YYYY-MM-DD format)"""

    if isinstance(date, str):
        try:
            date_parsed = pd.to_datetime(date, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date format not recognized: {date}. "
                "Please use 'YYYY-MM-DD' format."
            )
    else:
        raise TypeError(f"Date must be string, got {type(date)}")

    date_str = date_parsed.strftime("%Y%m%d")
    return date_parsed, date_str


def get_evaluation_file_paths(date_str, data_path, dataset="nowcast"):
    """
    Construct and validate file paths for evaluation data.

    Args:
        date_str: Date in YYYYMMDD format (target date for nowcast, issue date for forecast)
        data_path: Root data directory path
        dataset: "nowcast" or "forecast"

    Returns:
        For nowcast: Tuple of (weather_path1, weather_path2, windpower_path1, windpower_path2)
        For forecast: Tuple of (weather_path, windpower_path1, windpower_path2)

    Raises:
        FileNotFoundError: If required files don't exist
    """

    # Helper to build windpower file path
    def windpower_path(date_str):
        return os.path.join(
            data_path,
            "evaluation",
            "wind_power",
            f"wind_power_per_bidzone_{date_str}.parquet",
        )

    # Helper to validate file existence
    def validate_files(file_paths, dataset_name):
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{dataset_name} file not found: {path}")

    # Parse date and calculate target date
    parsed_date = pd.to_datetime(date_str, format="%Y%m%d")

    if dataset == "nowcast":
        # Nowcast: date_str is the target date we want to predict for
        # To get target_date 00:00-23:00, we need two files (11-hour shift)
        target_date = parsed_date
        date1 = target_date
        date2 = target_date + pd.Timedelta(days=1)

        date_str1 = date1.strftime("%Y%m%d")
        date_str2 = date2.strftime("%Y%m%d")

        # Build weather paths
        weather_folder = "met_nowcast_daily"
        path_to_weather1 = os.path.join(
            data_path, "evaluation", weather_folder, f"met_nowcast_{date_str1}.parquet"
        )
        path_to_weather2 = os.path.join(
            data_path, "evaluation", weather_folder, f"met_nowcast_{date_str2}.parquet"
        )

        # Build windpower paths
        path_to_windpower1 = windpower_path(date_str1)
        path_to_windpower2 = windpower_path(date_str2)

        # Validate all files exist
        validate_files(
            [
                path_to_weather1,
                path_to_weather2,
                path_to_windpower1,
                path_to_windpower2,
            ],
            f"Weather/windpower {dataset}",
        )

        return (
            path_to_weather1,
            path_to_weather2,
            path_to_windpower1,
            path_to_windpower2,
        )

    elif dataset == "forecast":
        # Forecast: date_str is issue date, predict for date+2
        target_date = parsed_date + pd.Timedelta(days=2)

        # Build weather path (single CSV file with all ensemble members)
        weather_folder = "met_forecast_daily"
        path_to_weather = os.path.join(
            data_path, "evaluation", weather_folder, f"meps_windai_{date_str}T09Z.csv"
        )

        # Build windpower paths (need two files for 11-hour shift)
        date_str1 = target_date.strftime("%Y%m%d")
        date_str2 = (target_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        path_to_windpower1 = windpower_path(date_str1)
        path_to_windpower2 = windpower_path(date_str2)

        # Validate all files exist
        validate_files(
            [path_to_weather, path_to_windpower1, path_to_windpower2],
            f"Weather/windpower {dataset}",
        )

        return path_to_weather, path_to_windpower1, path_to_windpower2

    else:
        raise ValueError(
            f"Unknown dataset type: {dataset}. Use 'nowcast' or 'forecast'"
        )


def check_evaluation_data_quality(weather_df, windpower_series, bid_zone):
    """Validate that evaluation data has no missing values"""
    # Check wind power
    if windpower_series.isna().any():
        nan_count = windpower_series.isna().sum()
        raise ValueError(
            f"Found {nan_count} NaN value(s) in wind power data for {bid_zone}. "
            "Evaluation data should not contain missing values."
        )

    # Check weather
    if weather_df.isna().any().any():
        nan_cols = weather_df.columns[weather_df.isna().any()].tolist()
        nan_counts = weather_df.isna().sum()[weather_df.isna().any()]
        raise ValueError(
            f"Found NaN values in weather data columns: {dict(zip(nan_cols, nan_counts))}. "
            "Evaluation data should not contain missing values."
        )


def load_evaluation_data(
    dates,
    scaler_params,
    site_scaler_params,
    bid_zone="ELSPOT NO1",
    dataset="nowcast",
    ensemble_member=0,
    data_path="../data",
):
    """
    Load and preprocess evaluation data for a specific dates.

    Same preprocessing pipeline as training data,
    but uses saved scaler parameters instead of fitting new ones. The function
    follows these steps:
    1. Parse dates
    2. Construct file paths and load raw data
    3. Transform features (wind direction)
    4. Filter and align data for bid zone
    5. Validate data quality (no NaN values)
    6. Apply training scaler parameters
    7. Convert to arrays
    """
    print(f"Loading evaluation data for {len(dates)} dates in zone {bid_zone}...")

    weather_data_period = []
    wind_data_period = []

    feature_cols, site_meta_cols, _ = get_feature_columns()

    # Load data for each date
    for date in dates:

        # 1. Parse date and get file paths
        date_parsed, date_str = parse_evaluation_date(date)
        print(
            f"Loading evaluation data for {date_parsed.strftime('%Y-%m-%d')} in zone {bid_zone}"
        )

        # 2. Construct file paths and load raw data
        windpower_paths = get_evaluation_file_paths(date_str, data_path, dataset)

        if dataset == "nowcast":
            (
                path_to_weather1,
                path_to_weather2,
                path_to_windpower1,
                path_to_windpower2,
            ) = windpower_paths

            # Load and combine two nowcast files (shifted by 11 hours)
            weather1 = load_weather_nowcast(path_to_weather1)
            weather2 = load_weather_nowcast(path_to_weather2)
            # Concatenate and sort by timestamp
            weather_data = pd.concat([weather1, weather2]).sort_index()

            # Load and combine two windpower files (shifted by 11 hours)
            windpower1 = load_windpower(path_to_windpower1)
            windpower2 = load_windpower(path_to_windpower2)
            # Concatenate and sort by timestamp
            windpower = pd.concat([windpower1, windpower2]).sort_index()

            # Filter to target day (00:00-23:00)
            target_date = pd.to_datetime(date_str, format="%Y%m%d")
            target_start = target_date.replace(hour=0, minute=0, second=0)
            target_end = target_date.replace(hour=23, minute=0, second=0)

            # Localize to CET to match data timezone
            target_start = target_start.tz_localize("CET")
            target_end = target_end.tz_localize("CET")

            # Filter weather to target day
            weather_data = weather_data[
                (weather_data.index >= target_start)
                & (weather_data.index <= target_end)
            ]
            weather_data = add_shifted_features(weather_data)

            weather_data = sin_cos_transform(
                weather_data, columns=["wind_direction_10m", "hour"]
            )

            # Filter windpower to target day
            windpower = windpower[
                (windpower.index >= target_start) & (windpower.index <= target_end)
            ]

        elif dataset == "forecast":  # forecast
            path_to_weather, path_to_windpower1, path_to_windpower2 = windpower_paths

            # Load forecast, rename columns, and select ensemble member
            weather_forecast = load_weather_forecast(path_to_weather)
            weather_forecast_renamed = rename_met_forecast_columns(weather_forecast)

            # ----variance calculation----
            # TODO put variance calulcation in a function
            # Wind direction transformation of all ensemble members to get correct variance
            weather_forecast_transformed = sin_cos_transform(
                weather_forecast_renamed, prefix="wind_direction_10m", n_members=15
            )
            # Calculate variance across ensemble members
            exclude = {
                "hour_x",
                "hour_y",
                "wind_speed_10m-1",
                "wind_speed_10m+1",
            }

            feature_member_cols = {
                feat: [f"{feat}_member_{i}" for i in range(15)]
                for feat in feature_cols
                if feat not in exclude
            }
            var_by_feature = {}
            for feat, cols in feature_member_cols.items():
                var_by_feature[feat] = (
                    weather_forecast_transformed.groupby("lead_time_hours")[cols]
                    .var()
                    .mean(axis=1)
                )
            weather_vars = pd.DataFrame(var_by_feature)
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            weather_vars_scaled = pd.DataFrame(
                scaler.fit_transform(weather_vars),
                index=weather_vars.index,
                columns=weather_vars.columns,
            )
            overall_variance = weather_vars_scaled.mean(axis=1)

            weather_forecast_transformed = weather_forecast_transformed.join(
                overall_variance.rename("variance"), on="lead_time_hours"
            )

            feature_cols += ["variance"]
            # ----variance calculation----

            weather_data = select_ensemble_member(
                weather_forecast_transformed, member=ensemble_member
            )
            # Set time as index to match nowcast format
            weather_data = weather_data.set_index("time")
            weather_data = add_shifted_features(weather_data)

            weather_data = sin_cos_transform(weather_data, columns=["hour"])

            # Load and combine two windpower files (shifted by 11 hours)
            windpower1 = load_windpower(path_to_windpower1)
            windpower2 = load_windpower(path_to_windpower2)
            # Concatenate and sort by timestamp
            windpower = pd.concat([windpower1, windpower2]).sort_index()

            # Filter to target prediction day (date + 2 days, 00:00-23:00)
            forecast_date = pd.to_datetime(date_str, format="%Y%m%d")
            target_date = forecast_date + pd.Timedelta(days=2)
            target_start = target_date.replace(hour=0, minute=0, second=0)
            target_end = target_date.replace(hour=23, minute=0, second=0)

            # Localize to CET to match data timezone
            target_start = target_start.tz_localize("CET")
            target_end = target_end.tz_localize("CET")

            # Filter weather forecast to target day
            weather_data = weather_data[
                (weather_data.index >= target_start)
                & (weather_data.index <= target_end)
            ]

            # Filter windpower to target day
            windpower = windpower[
                (windpower.index >= target_start) & (windpower.index <= target_end)
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Filtered to target day: {target_date.strftime('%Y-%m-%d')}")
        weather_data_period.append(weather_data)
        wind_data_period.append(windpower)

        weather_data = pd.concat(weather_data_period, axis=0)
        windpower = pd.concat(wind_data_period, axis=0)

    windparks = load_windparks(f"{data_path}/windparks_bidzone.csv")

    # 4. Filter and align data for specific bid zone, calculate capacity factor
    weather_multiindex, windpower_with_cf = prepare_bid_zone_data(
        weather_data, windpower, windparks, bid_zone
    )

    # 5. Validate data quality (no NaNs)
    windpower_series = windpower[bid_zone]
    check_evaluation_data_quality(weather_data, windpower_series, bid_zone)

    # 6. Extract features and apply scaling

    # Extract site metadata
    sites_meta_unscaled = (
        weather_multiindex[site_meta_cols]
        .drop_duplicates()
        .dropna()
        .sort_values("windpark_id")["operating_power_max"]
        .to_numpy()
    )

    # Select only feature columns
    weather_features = weather_multiindex[feature_cols]

    # Apply scaling using training scaler parameters
    weather_scaled = apply_scaler_params(weather_features, scaler_params)
    sites_meta, _ = scale_site_metadata(sites_meta_unscaled, site_scaler_params)

    # 7. Convert to arrays
    X_eval = get_input_shaped(weather_scaled, feature_cols)
    Y_eval = convert_targets_to_arrays(windpower_with_cf, prefix="Y_eval_")
    timestamps = weather_scaled.index.get_level_values("time").unique()

    print(
        f"Evaluation data shape: X={X_eval.shape}, Y_gen={Y_eval['Y_eval_gen'].shape}"
    )

    # Return combined results
    return {
        "X_eval": X_eval,
        **Y_eval,
        "sites_meta": sites_meta,
        "feature_cols": feature_cols,
        "index": timestamps,
    }


def rename_met_forecast_columns(df):
    """
    Rename MET Forecast columns to match MET Nowcasting variable naming.
    Handles ensemble members and drops columns not present in nowcast (e.g., gust).
    Handles both single-digit (_0, _1) and zero-padded (_00, _01) member formats.
    """

    renamed = {}
    columns_to_drop = []

    # Base map for variables without members
    base_map = {
        "sid": "windpark",
        "time": "time",
        "time_ref": "time_ref",
        "lt": "lead_time_hours",
    }

    # Direct mapping for shared variable names
    renamed.update(base_map)

    # Regex pattern to catch ensemble member columns like ws10m_3 or ws10m_03
    pattern = re.compile(r"^(?P<var>[a-zA-Z0-9_]+)_(?P<member>\d{1,2})$")

    for col in df.columns:
        if col in base_map:
            continue

        match = pattern.match(col)
        if match:
            var = match.group("var")
            member = match.group("member")

            # Normalize member number (remove leading zeros: "00" -> "0")
            member_normalized = str(int(member))

            # Map forecast variable prefix -> nowcast variable name
            if var == "ws10m":
                new = f"wind_speed_10m_member_{member_normalized}"

            elif var == "wd10m":
                new = f"wind_direction_10m_member_{member_normalized}"

            elif var == "t2m":
                new = f"air_temperature_2m_member_{member_normalized}"

            elif var == "mslp":
                new = f"air_pressure_at_sea_level_member_{member_normalized}"

            elif var == "rh2m":
                new = f"relative_humidity_2m_member_{member_normalized}"

            elif var == "g10m" or var.startswith(
                "gust"
            ):  # wind gust is not in nowcast, drop it
                columns_to_drop.append(col)
                continue

            else:
                print(f"Warning: Unknown forecast variable '{var}' in column '{col}'")
                continue

            renamed[col] = new
        else:
            if col not in base_map:
                print(f"Warning: Non-member column not mapped: '{col}'")

    # Apply renaming
    df_renamed = df.rename(columns=renamed)

    # Drop columns not in nowcast (like gust)
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} gust columns not present in nowcast")
        df_renamed = df_renamed.drop(columns=columns_to_drop, errors="ignore")

    print(f"\nColumn mapping summary:")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Renamed columns: {len(renamed)}")
    print(f"  Dropped columns: {len(columns_to_drop)}")
    print(f"  Final columns: {len(df_renamed.columns)}")

    return df_renamed


def select_ensemble_member(df, member=0):
    """Select a specific ensemble member from forecast data and rename to nowcast format"""
    # Identify columns for this member
    member_suffix = f"_member_{member}"

    # Columns to keep (non-member columns + this member's columns)
    non_member_cols = ["windpark", "time", "time_ref", "lead_time_hours", "variance"]
    member_cols = [col for col in df.columns if col.endswith(member_suffix)]

    # Check which non-member columns actually exist
    existing_non_member_cols = [col for col in non_member_cols if col in df.columns]
    missing_cols = [col for col in non_member_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")

    # Select columns
    selected_cols = existing_non_member_cols + member_cols
    df_member = df[selected_cols].copy()

    # Rename member columns to remove suffix (match nowcast naming)
    rename_dict = {col: col.replace(member_suffix, "") for col in member_cols}
    df_member = df_member.rename(columns=rename_dict)

    print(f"Selected ensemble member {member}")
    print(
        f"  Features: {[col for col in df_member.columns if col not in existing_non_member_cols]}"
    )

    return df_member
