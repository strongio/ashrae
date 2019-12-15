# +
import pandas as pd
import numpy as np
import torch

from sklearn.base import BaseEstimator, TransformerMixin

from torch.nn import ModuleDict, Embedding


# -

class DataFrameScaler(TransformerMixin, BaseEstimator):
    def __init__(self, value_colnames: list, group_colname: str = None, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale
        assert self.center or self.scale
        self.group_colname = group_colname
        self.value_colnames = tuple(value_colnames)
        self._fitted = None
        
    def fit(self, X: pd.DataFrame, y: None = None, **kwargs) -> pd.DataFrame:
        group_colname = self.group_colname or '_dummy'
        self._fitted = X.assign(_dummy = 1).groupby([group_colname])[self.value_colnames].agg(['mean','std']).reset_index()
        self._fitted.columns = [
            '__'.join(subcol for subcol in col if subcol != '').strip() 
            for col in self._fitted.columns.values
        ]
        return self
    
    def transform(self, X: pd.DataFrame, keep_cols: tuple=(), **kwargs) -> pd.DataFrame:
        group_colname = self.group_colname or '_dummy'
        if keep_cols == 'all':
            keep_cols = list(set(X.columns) - {group_colname} - set(self.value_colnames))
        out = X.\
            assign(_dummy = 1).\
            loc[:,(group_colname,) + tuple(keep_cols) + self.value_colnames].\
            merge(self._fitted, on=group_colname, how='left')
        for value_colname in self.value_colnames:
            if self.center:
                out[value_colname] -= out[f"{value_colname}__mean"]
            out.pop(f"{value_colname}__mean")
            if self.scale:
                out[value_colname] /= out[f"{value_colname}__std"]
            out.pop(f"{value_colname}__std")
        if '_dummy' in out.columns:
            out.pop('_dummy')
        return out


def date_expander(dataframe: pd.DataFrame,
                  start_dt_colname: str,
                  end_dt_colname: str,
                  time_unit: str,
                  new_colname: str,
                  end_inclusive: bool) -> pd.DataFrame:
    td = pd.Timedelta(1, time_unit)

    # add a timediff column:
    dataframe['_dt_diff'] = dataframe[end_dt_colname] - dataframe[start_dt_colname]

    # get the maximum timediff:
    max_diff = int((dataframe['_dt_diff'] / td).max())

    # for each possible timediff, get the intermediate time-differences:
    df_diffs = pd.concat([pd.DataFrame({'_to_add': np.arange(0, dt_diff + end_inclusive) * td}).assign(_dt_diff=dt_diff * td)
                          for dt_diff in range(max_diff + 1)])

    # join to the original dataframe
    data_expanded = dataframe.merge(df_diffs, on='_dt_diff')

    # the new dt column is just start plus the intermediate diffs:
    data_expanded[new_colname] = data_expanded[start_dt_colname] + data_expanded['_to_add']

    # remove start-end cols, as well as temp cols used for calculations:
    data_expanded = data_expanded.drop(columns=[start_dt_colname, end_dt_colname, '_to_add', '_dt_diff'])

    # don't modify dataframe in place:
    del dataframe['_dt_diff']

    return data_expanded


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
        assert isinstance(delete_interval, Timedelta)
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


def clean_readings(df: pd.DataFrame, 
                   roll_days: tuple = (3,14), 
                   multis: tuple = (0.66, 1.33),
                   group_colname: str = 'ts_id', 
                   value_colname: str = 'meter_reading',
                   std_dev_theshold: float = .01):
    df = df.copy()
    
    # running sd:
    df['rolling_std'] = df.groupby(group_colname)[value_colname].\
        rolling(roll_days[0] * 24, center=True).std().reset_index(0, drop=True)
    df[f'{value_colname}_clean'] = df[value_colname]
    df.loc[df['rolling_std'] <= std_dev_theshold, f'{value_colname}_clean'] = np.nan

    # running lower bound:
    # two-pass to deal w/tradeoff between avoiding false-alarms on meters like 256,
    # while not letting ones like 1269 through
    # (living with mediocre cleaning for: 746)
    df['_lb'] = df.groupby(group_colname)[f'{value_colname}_clean'].rolling(24, center=True).min().reset_index(0, drop=True)
    p1 = df.groupby(group_colname)['_lb'].\
        rolling(roll_days[1] * 24, center=True).median().reset_index(0, drop=True)
    p2 = df.groupby(group_colname)['_lb'].\
        rolling(roll_days[0] * 24, center=True).median().reset_index(0, drop=True)
    df['_lb_roll'] = (p1 + p2) / 2
    df['_lb_roll'].fillna(df.groupby([group_colname])['_lb_roll'].transform('min'), inplace=True)

    # running upper bound:
    df['_ub'] = df.groupby(group_colname)[f'{value_colname}_clean'].\
        rolling(24, center=True).max().reset_index(0, drop=True)
    df['_ub_roll'] = df.groupby(group_colname)['_ub'].\
        rolling(roll_days[1] * 24, center=True).median().reset_index(0, drop=True)
    df['_ub_roll'].fillna(df.groupby([group_colname])['_ub_roll'].transform('max'), inplace=True)

    # clean:
    df['lower_thresh'] = df['_lb_roll'] * multis[0]
    df.loc[df[f'{value_colname}_clean'] <= df['lower_thresh'], f'{value_colname}_clean'] = np.nan
    bound_diff = df['_ub_roll'] - df['_lb_roll']
    df['upper_thresh'] = df['_ub_roll'] + multis[1] * bound_diff
    df.loc[df[f'{value_colname}_clean'] >= df['upper_thresh'], f'{value_colname}_clean'] = np.nan
    
    del df['_lb']
    del df['_lb_roll']
    del df['_ub']
    del df['_ub_roll']
    
    return df


class TimeSeriesStateNN(torch.nn.Module):
    def __init__(self, embed_inputs: dict, sub_module: torch.nn.Module):
        super().__init__()
        self.sub_module = sub_module
        self.embed_modules = ModuleDict()
        for idx, kwargs in embed_inputs.items():
            self.embed_modules[str(idx)] = Embedding(**kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sub_module_idx = list(range(input.shape[-1]))
        to_concat = [None]
        for idx_str, embed_module in self.embed_modules.items():
            idx = int(idx_str)
            to_concat.append(embed_module(input[..., idx].to(torch.long)))
            sub_module_idx.remove(idx)
        to_concat[0] = input[..., sub_module_idx]
        return self.sub_module(torch.cat(to_concat, dim=-1))
    
    @property
    def out_features(self):
        return self.sub_module[-1].out_features


def forward_backward(model: 'KalmanFilter',
                     batch: 'TimeSeriesDataset',
                     **kwargs) -> 'TimeSeriesDataset':
    batch = remove_random_dates(batch, which=0, **kwargs)
    readings, predictors = batch.tensors
    prediction = model(
        readings,
        start_datetimes=batch.start_datetimes,
        predictors=predictors,
        progress=True
    )
    lp = prediction.log_prob(readings)
    loss = -lp.mean()
    if loss.requires_grad:
        loss.backward()
        model.optimizer.step()
    return loss


# +
try:
    from plotnine import *
except ImportError:
    from fake_plotnine import *
    
def loss_plot(df_loss):
    return (
        ggplot(pd.DataFrame(df_loss), aes(x='epoch', y='value')) + 
            stat_summary(fun_y=np.mean, geom='line') + facet_wrap("~dataset", scales='free') +
            theme_bw() + theme(figure_size=(10,4)) 
        )
