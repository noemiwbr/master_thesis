# This file was copied from EquiWindNerds
import logging
import os
import re
from functools import cache

import Levenshtein as lv
import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm
import sys


def load_maintenance_events(
    maintenance_events_dir="../data/UMM_Messages.xlsx",
) -> pd.DataFrame:
    """load maintenance events"""

    # read raw excel file
    maintenance_events_download_instructions = "Either replace with API calls to Nord Poll UMM Messages or visit website, filter to 'planned', 'onshore wind', 'NO1'-'NO5' (incl A suffix) and download"
    if not os.path.exists(maintenance_events_dir):
        raise FileNotFoundError(
            f"File not found at {maintenance_events_dir}. {maintenance_events_download_instructions}"
        )
    maintenance_events = pd.read_excel(maintenance_events_dir, skiprows=3)
    logging.warning(
        f"Loading a static dataset. {maintenance_events_download_instructions}"
    )
    return maintenance_events


def process_maintenance_events(maintenance_events: pd.DataFrame) -> pd.DataFrame:
    # drop columns containing information that is a duplicate, low cardinality or low impact
    maintenance_events.drop(
        columns=["Duration", "Fuel Type", "Assets", "Revision", "Event"], inplace=True
    )

    # convert time columns
    umm_format = "%Y-%m.%dT%H:%M"
    maintenance_events["Event Start"] = pd.to_datetime(
        maintenance_events["Event Start"], format=umm_format, utc=True
    ).dt.tz_convert("CET")
    maintenance_events["Event Stop"] = pd.to_datetime(
        maintenance_events["Event Stop"], format=umm_format, utc=True
    ).dt.tz_convert("CET")
    maintenance_events["Event Duration"] = (
        (
            maintenance_events["Event Stop"] - maintenance_events["Event Start"]
        ).dt.total_seconds()
        / 3600
        / 24
    )
    maintenance_events["Published"] = pd.to_datetime(
        maintenance_events["Published"], format=umm_format + ":%S", utc=True
    ).dt.tz_convert("CET")

    # drop dismissed maintenance
    maintenance_events = maintenance_events.loc[
        maintenance_events["Status"] != "Dismissed"
    ]
    assert maintenance_events["Status"].nunique() == 1
    maintenance_events.drop(
        columns=["Remarks", "Status", "Cancellation Reason"], inplace=True
    )

    # drop instances where there was no unavailability
    maintenance_events = maintenance_events.loc[maintenance_events["Unavailable"] > 0]

    # calculate total installed capacity for improved joining
    maintenance_events["Total"] = (
        maintenance_events["Available"] + maintenance_events["Unavailable"]
    )

    return maintenance_events


def load_wind_park_metadata():
    """load wind park metadata"""

    # load reference wind park metadata
    wind_park_metadata = pd.read_csv("../data/windparks_bidzone.csv")
    wind_park_metadata.drop(columns="eic_code", inplace=True)
    wind_park_metadata.drop_duplicates(inplace=True)
    wind_park_metadata["prod_start_new"] = pd.to_datetime(
        wind_park_metadata["prod_start_new"], utc=True
    ).dt.tz_convert("CET")

    return wind_park_metadata


def name_clean(x):
    """clean and simplify wind park names"""

    x = x.lower()
    x = x.replace("vindkrv", "")
    x = x.replace("wind", "")
    x = x.replace("vind", "")
    x = x.replace("park", "")
    x = x.replace("farm", "")
    x = x.replace("kraft", "")
    x = x.replace("verk", "")
    x = x.replace("norsk", "")
    x = x.replace("norks", "")
    x = x.replace(" a/s", " ")
    x = x.replace(" as", " ")
    x = x.replace("-", " ")
    x = x.replace("tromsø", "kvitfjell")
    x = x.replace("kvittfjell", "kvitfjell")
    x = x.replace("kvitfjell", "", x.count("kvitfjell") - 1)
    x = x.replace("bessakerfjellet", "bessaker")
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .")
    return x


class CapacityFactorTransformer(BaseEstimator, TransformerMixin):
    """custom transformer to convert power to/from capacity factor"""

    def __init__(self, maintenance_events=pd.DataFrame()):
        # load raw datasets
        if maintenance_events.empty:
            maintenance_events = load_maintenance_events()
        self.maintenance_events = process_maintenance_events(maintenance_events)
        self.wind_park_metadata = load_wind_park_metadata()

        # identify correct windpark name and area from reference metadata
        self.maintenance_events["Windpark_"] = self.maintenance_events[
            "Infrastructure"
        ].map(lambda x: x.split("|")[1].strip())
        self.maintenance_events["Windpark"] = self.maintenance_events[
            ["Total", "Windpark_"]
        ].apply(
            lambda x: self.windpark_name_alignment(
                name=x["Windpark_"], capacity=x["Total"]
            ),
            axis=1,
        )
        self.maintenance_events["Area"] = self.maintenance_events["Windpark"].map(
            self.wind_park_metadata.set_index("substation_name")["bidding_area"]
        )
        self.maintenance_events.drop(
            columns=["Infrastructure", "Total", "Windpark_"], inplace=True
        )

        # calculate available installed capacity across full time range and all biddings area
        date_range = pd.date_range(
            start=self.wind_park_metadata["prod_start_new"].min(),
            end=self.maintenance_events["Event Stop"].max(),
            freq="h",
        )
        available_dicts = pqdm(
            date_range,
            self.calculate_available_capacity,
            n_jobs=int(np.ceil(os.cpu_count() * 0.75)),
            desc="Calculating available capacity across full time range",
            disable=not sys.stdout.isatty(),
        )
        self.available = pd.DataFrame(available_dicts, index=date_range)

    @cache
    def windpark_name_alignment(self, name, capacity):
        """identify windpark name from reference metadata"""

        # initialise incumbents
        sim_best = 0
        name_best = None

        # loop over all possible reference wind farms
        for _, comp in self.wind_park_metadata.iterrows():
            # extract comparison name and capacity
            name_comp = comp["substation_name"]
            capacity_comp = comp["operating_power_max"]

            # calculate similarity to comparison name
            name_sim_denom = min(len(name), len(name_comp))
            name_sim_num = min(
                lv.distance(name, name_comp, weights=(1, 0.5, 1), processor=name_clean),
                name_sim_denom,
            )
            name_sim = 1 - name_sim_num / name_sim_denom

            # calculate similarity to comparison capacity
            cap = 1 - np.clip(
                abs(capacity - capacity_comp) / min(capacity, capacity_comp), 0, 1
            )

            # calculate voerall similarity score
            sim = (name_sim + cap) / 2

            # update incumbent
            if sim > sim_best:
                name_best = name_comp
                capacity_best = capacity_comp
                sim_best = sim

        # fix/log failed matches
        if sim_best < 0.85:
            if name == "Øyfjellet":
                name_best, sim_best = "Øyfjell1", None
                logging.info(
                    f"No match found for {name} ({capacity} MW). Manually mapping to {name_best}"
                )
            elif name == "Songkjølen and Engerfjellet":
                name_best, sim_best = "Songkjølen", None
                logging.info(
                    f"No match found for {name} ({capacity} MW). Manually mapping to {name_best}"
                )
            elif name == "Hitra II":
                name_best, sim_best = "Hitra", None
                logging.info(
                    f"No match found for {name} ({capacity} MW). Manually mapping to {name_best}"
                )
            elif name in ["Bjerkreim Vind", "Bjerkreim Vind AS"]:
                name_best, sim_best = "Bjerk_VK Vindpark", None
                logging.info(
                    f"No match found for {name} ({capacity} MW). Manually mapping to {name_best}"
                )
            else:
                logging.warning(
                    f"No match found for {name} ({capacity} MW). Closest match ({sim_best:.3f}) found was {name_best} ({capacity_best} MW)"
                )
                name_best, sim_best = None, None

        return name_best

    def calculate_available_capacity(self, datetime):
        """calculate available installed capacity"""

        # installed capacity at time provided
        installed = (
            self.wind_park_metadata.loc[
                self.wind_park_metadata["prod_start_new"] <= datetime
            ]
            .groupby("bidding_area")["operating_power_max"]
            .sum()
        )

        # reduction of availbility due to maintenance
        unavailable = (
            self.maintenance_events.loc[
                (datetime >= self.maintenance_events["Event Start"])
                & (datetime <= self.maintenance_events["Event Stop"])
            ]
            .groupby("Area")["Unavailable"]
            .sum()
        )

        # installed capacity that is availble
        available = installed.sub(unavailable, fill_value=0).clip(lower=0)

        return available.to_dict()

    def available_subset(self, X):
        return self.available.loc[X.index]

    def fit(self, X, y=None):
        return self

    def transform(self, X, bid_zone):
        available_power = self.available_subset(X)
        return X / available_power[bid_zone]

    def inverse_transform(self, X_trans, bid_zone):
        available_power = self.available_subset(X_trans)
        return X_trans * available_power[bid_zone]
