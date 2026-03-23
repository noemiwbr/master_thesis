import numpy as np


def unstandardize(data, split):
    scaler = data["scaler_params"]
    feature_cols = data["feature_cols"]
    unstandardized = data.copy()
    unstandardized[split] = data[split].copy()
    for key in scaler["mean"].keys():
        i = feature_cols.index(key)
        # Forecast
        if len(unstandardized[split].shape) == 4:
            unstandardized[split][:, :, :, i] = (
                data[split][:, :, :, i] * scaler["std"][key] + scaler["mean"][key]
            )
        # Nowcast
        else:
            unstandardized[split][:, :, i] = (
                data[split][:, :, i] * scaler["std"][key] + scaler["mean"][key]
            )
    return unstandardized


def air_density(data, split):
    temperature_idx = data["feature_cols"].index("air_temperature_2m")
    air_pressure_idx = data["feature_cols"].index("air_pressure_at_sea_level")
    humidity_idx = data["feature_cols"].index("relative_humidity_2m")

    if len(data[split].shape) == 4:
        temperature = data[split][:, :, :, temperature_idx]
        air_pressure = data[split][:, :, :, air_pressure_idx]
        humidity = data[split][:, :, :, humidity_idx]
    else:
        temperature = data[split][:, :, temperature_idx]
        air_pressure = data[split][:, :, air_pressure_idx]
        humidity = data[split][:, :, humidity_idx]

    rho = (air_pressure / (287.05 * temperature)) * (1 - 0.1 * humidity)

    return rho


def wind_power_curve_component(
    data,
    split="X_train",
    cut_in=3.0,
    rated=12.0,
    cut_out=25.0,
    rated_power_mw=3,
):

    speed_idx = data["feature_cols"].index("wind_speed_10m")
    if len(data[split].shape) == 4:
        speed = data[split][:, :, :, speed_idx]
    else:
        speed = data[split][:, :, speed_idx]

    rho = air_density(data, split=split)
    rho0 = 1.225

    # before cut-in and after cut-out
    power = np.zeros_like(speed, dtype=float)

    # S curve
    ramp_mask = (speed >= cut_in) & (speed < rated)
    v = speed[ramp_mask]
    rho_rel = rho[ramp_mask] / rho0

    ramp = (v**3 - cut_in**3) / (rated**3 - cut_in**3)
    ramp = np.clip(ramp, 0, 1)

    power[ramp_mask] = rated_power_mw * ramp * rho_rel

    # rated
    rated_mask = (speed >= rated) & (speed < cut_out)
    power[rated_mask] = rated_power_mw
    return power


def wind_direction_component(data, split="X_train", alpha=0.1, preferred_dir=270):

    x_idx = data["feature_cols"].index("wind_direction_10m_x")
    y_idx = data["feature_cols"].index("wind_direction_10m_y")

    if len(data[split].shape) == 4:
        x = data[split][:, :, :, x_idx]
        y = data[split][:, :, :, y_idx]
    else:
        x = data[split][:, :, x_idx]
        y = data[split][:, :, y_idx]

    theta0 = np.deg2rad(preferred_dir)

    pref_x = np.cos(theta0)
    pref_y = np.sin(theta0)

    # Dot product
    factor = 1 + alpha * (x * pref_x + y * pref_y)

    return factor


def humidity_factor(data, split="X_train"):
    humidity_idx = data["feature_cols"].index("relative_humidity_2m")
    if len(data[split].shape) == 4:
        humidity = data[split][:, :, :, humidity_idx]
    else:
        humidity = data[split][:, :, humidity_idx]

    return 1 - 0.05 * humidity


def synthetic_power_data(data, split="X_train"):
    unstandardized = unstandardize(data, split)
    synt_power = wind_power_curve_component(
        unstandardized, split, rated_power_mw=1, cut_in=3
    ) * wind_direction_component(unstandardized, split, alpha=0.5)

    total_mw_bz = (
        data["sites_meta"] * data["site_scaler_params"]["std"]
        + data["site_scaler_params"]["mean"]
    ).sum()

    num_parks_total = total_mw_bz
    num_parks = (num_parks_total / unstandardized[split].shape[1]).round()

    synt_power += np.random.normal(0, 0.01, size=synt_power.shape)
    synt_power = np.clip(synt_power, 1e-4, 1)

    synthetic_power = (synt_power * num_parks).sum(1)

    print(synthetic_power.min(), synthetic_power.max())

    return synthetic_power, synt_power
