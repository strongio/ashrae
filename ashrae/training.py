import numpy as np
import torch
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.utils.data import TimeSeriesDataset


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


def forward_backward(model: 'KalmanFilter',
                     batch: 'TimeSeriesDataset',
                     **kwargs) -> 'TimeSeriesDataset':
    batch = remove_random_dates(batch, which=0, **kwargs)
    readings, predictors = batch.tensors
    prediction = model(readings, start_datetimes=batch.start_datetimes, predictors=predictors)
    lp = prediction.log_prob(readings)
    loss = -lp.mean()
    if loss.requires_grad:
        loss.backward()
        model.optimizer.step()
    return loss
