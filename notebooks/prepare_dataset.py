# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

from torch_kalman.utils.data import TimeSeriesDataset
from torch_kalman.utils.features import fourier_model_mat

from utils import DataFrameScaler, date_expander

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from IPython.display import clear_output

import torch
# -

season_config = {
    'season_start' : pd.Timestamp('2007-01-01'),  # arbitrary monday at midnight
    'dt_unit' : 'h'
}
colname_config = {
    'group_colname':'building_id',
    'time_colname':'timestamp'
}

# ## Weather Data

print("Processing weather data...")

# +
df_weather_train = pd.read_csv("../data/weather_train.csv", parse_dates=['timestamp'])

def get_rel_humidity(air_temp: np.ndarray, dew_temp: np.ndarray) -> np.ndarray:
    numer = np.exp( (17.625 * dew_temp) / (243.04 + dew_temp) )
    denom = np.exp( (17.625 * air_temp) / (243.04 + air_temp) )
    return numer / denom

df_weather_train['relative_humidity'] =\
    get_rel_humidity(df_weather_train['air_temperature'], df_weather_train.pop('dew_temperature'))
df_weather_train['is_raining'] = (df_weather_train['precip_depth_1_hr'].fillna(0.0) > 0.0).astype('int')

# +
weather_preds = [
    'air_temperature', 'relative_humidity', 'is_raining', 'sea_level_pressure', 'wind_speed', 'cloud_coverage'
]
weather_scaler = DataFrameScaler(value_colnames=weather_preds).fit(df_weather_train)
df_weather_train_pp = weather_scaler.transform(df_weather_train, keep_cols=('site_id','timestamp'))
df_weather_train_pp['air_temperature_hi'] =\
    df_weather_train_pp['air_temperature'].where(df_weather_train_pp['air_temperature'] > 0, 0.0)
df_weather_train_pp['air_temperature_lo'] =\
    df_weather_train_pp['air_temperature'].where(df_weather_train_pp.pop('air_temperature') <= 0, 0.0)

weather_preds.remove('air_temperature')
weather_preds.extend(['air_temperature_hi','air_temperature_lo'])
# -

print("... finished.")

# ## External Predictors 

print("Processing external predictors...")

# +
df_meta_predictors = pd.read_csv("../data/building_metadata.csv").\
    assign(primary_use=lambda df: df['primary_use'].astype('category').cat.codes)

meta_preds = ['year_built']

df_meta_predictors['floor_count_sqrt'] = np.sqrt(df_meta_predictors['floor_count'].fillna(1))
meta_preds.append('floor_count_sqrt')

df_meta_predictors['square_feet_log10'] = np.log10(df_meta_predictors['square_feet'])
meta_preds.append('square_feet_log10')

meta_scaler = DataFrameScaler(meta_preds).fit(df_meta_predictors)
df_meta_predictors = meta_scaler.transform(df_meta_predictors, keep_cols=['site_id', 'building_id', 'primary_use'])

meta_preds.append('primary_use')
primary_uses = df_meta_predictors['primary_use'].unique()
# -

df_holidays = calendar().\
    holidays(start=df_weather_train_pp['timestamp'].min(), end=pd.Timestamp.today(), return_name=True).\
    reset_index().\
    rename(columns={0 : 'holiday', 'index' : 'start'}).\
    assign(end= lambda df: df['start'] + pd.Timedelta('1D'),
           holiday = lambda df: df['holiday'].astype('category')).\
    pipe(date_expander,
        start_dt_colname='start',
        end_dt_colname='end',
        time_unit='h',
        new_colname='timestamp',
        end_inclusive=False)
holidays = list(df_holidays['holiday'].cat.categories)
df_holidays['holiday'].cat.add_categories(['None'], inplace=True)
df_holidays['holiday'].cat.reorder_categories(new_categories=['None'] + holidays, inplace=True)

# +
df_tv_predictors_train = df_weather_train_pp.\
    merge(df_holidays, how='left', on=['timestamp']).\
    assign(
        is_weekday=lambda df: (df['timestamp'].dt.weekday < 5).astype('int'),
        holiday=lambda df: df['holiday'].fillna("None").cat.codes
    ).\
    reset_index(drop=True)

# df_tv_predictors_train['hour_in_day'] = df_tv_predictors_train['timestamp'].dt.hour
df_tv_predictors_train = pd.concat([
    df_tv_predictors_train, 
    pd.get_dummies(df_tv_predictors_train['timestamp'].dt.hour, prefix='hour', drop_first=True)
], 
    axis=1)

df_tv_predictors_train = pd.concat([
    df_tv_predictors_train, 
    fourier_model_mat(
        dt=df_tv_predictors_train['timestamp'], 
        K=4,
        period='yearly',
        start_dt=season_config['season_start'].to_datetime64(),
        output_dataframe=True
    )
    ],
    axis=1
)

tv_preds = [c for c in df_tv_predictors_train.columns if c not in ['building_id','timestamp','site_id']]
# -

print("...finished")


def prepare_dataset(df_readings: pd.DataFrame, reading_colname: str) -> TimeSeriesDataset:
    # join:
    df_joined = df_meta_predictors.\
        merge(df_tv_predictors_train, on=['site_id'], how='inner').\
        merge(df_readings, on=['building_id', 'timestamp'], how='left').\
        fillna({c : 0.0 for c in (meta_preds + tv_preds)})
    
    # filter out dates before the building started:
    _min_dts = df_joined.query(f"~{reading_colname}.isnull()").groupby('building_id')['timestamp'].min().to_dict()
    df_joined = df_joined.\
        assign(_min_dt = lambda df: df['building_id'].map(_min_dts)).\
        query("timestamp >= _min_dt")
    
    # create dataset:
    dataset = TimeSeriesDataset.from_dataframe(
                df_joined,
                **colname_config,
                measure_colnames=[reading_colname] + tv_preds + meta_preds,
                dt_unit='h'
            )
    # split into separate tensors for readings, preds:
    return dataset.split_measures(slice(1), slice(1,None))
