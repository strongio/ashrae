import numpy as np
import torch
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.utils.data import TimeSeriesDataset, TimeSeriesDataLoader

import pandas as pd


class DataLoaderFactory:
    def __init__(self,
                 df_meta_preds: pd.DataFrame,
                 df_tv_preds: pd.DataFrame,
                 predictors: list,
                 group_colname: str = 'building_id',
                 time_colname: str = 'timestamp'
                 ):
        self.df_meta_preds = df_meta_preds
        self.df_tv_preds = df_tv_preds
        self.predictors = list(predictors)
        assert group_colname == 'building_id'
        assert time_colname == 'timestamp'
        self.colnames = {'group_colname': group_colname, 'time_colname': time_colname}

    def __call__(self, df_readings: pd.DataFrame, reading_colname: str, **kwargs) -> 'TimeSeriesDataLoader':
        # filter out buildings not in readings:
        buildings = df_readings['building_id'].unique()
        df_meta_preds = self.df_meta_preds.loc[self.df_meta_preds['building_id'].isin(buildings), :]

        # filter by max date of readings:
        max_dt = df_readings['timestamp'].max()
        df_tv_preds = self.df_tv_preds.loc[self.df_tv_preds['timestamp'] <= max_dt,:]

        # join:
        df_joined = df_meta_preds. \
            merge(df_tv_preds, on=['site_id'], how='inner'). \
            merge(df_readings, on=['building_id', 'timestamp'], how='left'). \
            fillna({c: 0.0 for c in self.predictors})

        # filter out dates before the building started:
        df_joined = df_joined. \
            merge(
            df_joined.loc[~df_joined[reading_colname].isnull(), :]. \
                groupby('building_id'). \
                agg(_min_dt=('timestamp', 'min')). \
                reset_index()
        ). \
            query("timestamp >= _min_dt")

        # create dataloader:
        dataloader = TimeSeriesDataLoader.from_dataframe(
            df_joined,
            **self.colnames,
            measure_colnames=[reading_colname] + self.predictors,
            dt_unit='h',
            **kwargs
        )
        return dataloader


def remove_random_dates(batch: 'TimeSeriesDataset',
                        delete_interval: str,
                        which: 0,
                        keep_frac: float = .33,
                        random_state: np.random.RandomState = None
                        ) -> 'TimeSeriesDataset':
    """
    Delete chunks of data from a time-series by date. Useful for training models, so that they are "incentivized" to care
    about long-range forecasts rather than just one-step ahead forecasts.
    """
    # take the start-datetimes, convert to date-per-time (per-group), round to interval so that deletes are 'runs'
    # (e.g. we have hourly data but want to delete random days)

    if random_state is None:
        random_state = np.random.RandomState()

    if isinstance(delete_interval, str):
        delete_interval = np.timedelta64(1, delete_interval)
    else:
        assert isinstance(delete_interval, pd.Timedelta)
    dt_unit = np.timedelta64(1, batch.dt_unit)
    assert delete_interval >= dt_unit

    tens = batch.tensors[which]
    num_groups, num_timesteps, _ = tens.shape

    num_units_in_delete_unit = int(delete_interval / dt_unit)
    offset = random_state.choice(range(num_units_in_delete_unit), size=1)

    time_as_delete_unit = np.floor(np.arange(offset, num_timesteps + offset) / num_units_in_delete_unit)
    time_as_delete_unit_unique = np.unique(time_as_delete_unit)
    delete_times = random_state.choice(time_as_delete_unit_unique,
                                       size=int(len(time_as_delete_unit_unique) * (1 - keep_frac)),
                                       replace=False)

    delete_bool = np.broadcast_to(np.isin(time_as_delete_unit, delete_times), (num_groups, num_timesteps))
    keep_idx = np.where(~delete_bool)

    # create new tensor, only assign at keep-idx
    with_missing = torch.empty_like(tens)
    with_missing[:] = np.nan
    with_missing[keep_idx] = tens[keep_idx]
    new_tensors = list(batch.tensors)
    new_tensors[which] = with_missing
    return batch.with_new_tensors(*new_tensors)



