import pandas as pd

def load_weather_forecast(path: str):
    df = pd.read_parquet(path)
    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('CET')
    df['time_ref'] = df['time_ref'].dt.tz_localize('UTC').dt.tz_convert('CET')
    return df

def load_weather_nowcast(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize('UTC').tz_convert('CET')

def load_windpower(path: str):
    df = pd.read_parquet(path)
    return df.tz_localize('UTC').tz_convert('CET')

def load_windparks(path: str):
    return pd.read_csv(path)
