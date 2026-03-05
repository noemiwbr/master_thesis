import pandas as pd
import numpy as np
from data_processing.downtime import CapacityFactorTransformer


def filter_bid_zone(windparks, bid_zone: str):
    # Agg because windparks can have several eic_codes
    windparks_in_bid_zone = windparks[windparks["bidding_area"] == bid_zone]
    return windparks_in_bid_zone.groupby("substation_name", as_index=False).agg(
        {"operating_power_max": "max", "prod_start_new": "max"}
    )


def align_windpower_and_weather(windpower, weather):
    # Restrict to timestamps available in both datasets
    common_ts = weather.index.intersection(windpower.index)
    weather = weather.loc[common_ts]
    windpower = windpower.loc[common_ts].sort_index()
    return weather, windpower


def merge_metadata(weather_in_bid_zone, windparks_in_bid_zone):
    # Merge weather data with meta information (operating_power_max, prod_start_new)

    valid_windparks = windparks_in_bid_zone["substation_name"].unique()

    filtered_weather = weather_in_bid_zone[
        weather_in_bid_zone["windpark"].isin(valid_windparks)
    ]

    result = filtered_weather.merge(
        windparks_in_bid_zone.reset_index().rename(columns={"index": "windpark_id"})[
            [
                "substation_name",
                "windpark_id",
                "operating_power_max",
                "prod_start_new",
            ]
        ],
        left_on="windpark",
        right_on="substation_name",
        how="left",
    ).set_index(filtered_weather.index)
    return result


# TODO die Funktion macht gar nicht was sie soll
def multiindex(weather_in_bid_zone, windparks_in_bid_zone):
    weather_in_bid_zone.index.name = "time"
    weather_in_bid_zone = weather_in_bid_zone.set_index(
        "windpark", append=True
    ).sort_index()
    # Fill missing windpark–time combinations (full multi index) to include all combinations
    # TODO is this necessary??
    # all_times = weather_in_bid_zone.index.get_level_values('time').unique()
    # all_windparks = windparks_in_bid_zone['substation_name'].unique()
    # full_index = pd.MultiIndex.from_product([all_times, all_windparks], names=['time', 'windpark'])

    # weather_in_bid_zone = weather_in_bid_zone.reindex(full_index)

    # Set operating_power_max = NaN for windparks not yet operational
    for _, row in windparks_in_bid_zone.iterrows():
        windpark = row["substation_name"]
        start_date = row["prod_start_new"]
        mask = (weather_in_bid_zone.index.get_level_values("time") < start_date) & (
            weather_in_bid_zone.index.get_level_values("windpark") == windpark
        )
        # All timestamps before production start -> NaN
        weather_in_bid_zone.loc[mask, "operating_power_max"] = np.nan
    return weather_in_bid_zone.sort_index()


def capacity_factor(bid_zone, windpower_in_bid_zone):
    capacity_factor_transformer = CapacityFactorTransformer()
    gen = windpower_in_bid_zone.to_frame("generation")["generation"]
    avail = capacity_factor_transformer.available[bid_zone].reindex(gen.index)
    cf = gen / avail

    maintenance_events = capacity_factor_transformer.maintenance_events.copy()
    maintenance_events = maintenance_events[
        maintenance_events["Area"] == bid_zone
    ].dropna(subset=["Area", "Event Start", "Event Stop"])

    def hourly_range_inclusive(start, stop):
        return pd.date_range(start, stop, freq="h", inclusive="both")

    maintenance_events["time"] = maintenance_events.apply(
        lambda r: hourly_range_inclusive(r["Event Start"], r["Event Stop"]), axis=1
    )

    maintenance_hours_long = (
        maintenance_events[["time"]]
        .explode("time")
        .drop_duplicates()
        .loc[lambda df: df["time"].isin(windpower_in_bid_zone.index)]
        .sort_values(["time"])
        .reset_index(drop=True)
    )

    maintenance_hours = maintenance_hours_long["time"].nunique()
    # total_hours = len(windpower_in_bid_zone.index)
    # fraction = maintenance_hours / total_hours

    print(f"Maintenance hours in operational period: {maintenance_hours}")

    return (
        pd.DataFrame(
            {"generation": gen, "available_capacity": avail, "capacity_factor": cf}
        ),
        maintenance_hours_long,
    )


def split_train_test_meta(
    weather_in_bid_zone, windpower_in_bid_zone, feature_cols, site_meta_cols, split_time
):

    # TODO use inverted masks for split
    train_mask_power = windpower_in_bid_zone.index < split_time
    test_mask_power = windpower_in_bid_zone.index >= split_time
    train_mask_weather = weather_in_bid_zone.index.get_level_values("time") < split_time
    test_mask_weather = weather_in_bid_zone.index.get_level_values("time") >= split_time

    df_train_X = weather_in_bid_zone.loc[train_mask_weather, feature_cols].copy()
    df_test_X = weather_in_bid_zone.loc[test_mask_weather, feature_cols].copy()

    df_train_Y = windpower_in_bid_zone.loc[train_mask_power]
    df_test_Y = windpower_in_bid_zone.loc[test_mask_power]

    sites_meta = (
        weather_in_bid_zone[site_meta_cols]
        .drop_duplicates()
        .dropna()
        .sort_values("windpark_id")["operating_power_max"]
        .to_numpy()
    )

    return df_train_X, df_train_Y, df_test_X, df_test_Y, sites_meta


import torch


def to_tensor(data):
    tensor_data = {}
    keep = {
        "feature_cols",
        "scaler_params",
        "site_scaler_params",
        "X_train_timestamps",
        "X_test_timestamps",
    }

    for key, value in data.items():
        if key in keep:
            tensor_data[key] = value
        elif value is None:
            tensor_data[key] = None
        else:
            t = torch.as_tensor(value, dtype=torch.float32)
            tensor_data[key] = t

    return tensor_data
