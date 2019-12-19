import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class DataFrameScaler(TransformerMixin, BaseEstimator):
    def __init__(self, value_colnames: list, group_colname: str = None, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale
        assert self.center or self.scale
        self.group_colname = group_colname
        self.value_colnames = tuple(value_colnames)
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: None = None, **kwargs) -> 'DataFrameScaler':
        group_colname = self.group_colname or '_dummy'
        self._fitted = X.assign(_dummy=1).groupby([group_colname])[self.value_colnames].agg(
            ['mean', 'std']).reset_index()
        self._fitted.columns = [
            '__'.join(subcol for subcol in col if subcol != '').strip()
            for col in self._fitted.columns.values
        ]
        return self

    def _merge(self, X: pd.DataFrame, keep_cols: tuple) -> pd.DataFrame:
        group_colname = self.group_colname or '_dummy'
        if keep_cols == 'all':
            keep_cols = list(set(X.columns) - {group_colname} - set(self.value_colnames))
        out = X. \
                  assign(_dummy=1). \
                  loc[:, (group_colname,) + tuple(keep_cols) + self.value_colnames]. \
            merge(self._fitted, on=group_colname, how='left')
        return out

    def transform(self, X: pd.DataFrame, keep_cols: tuple = (), **kwargs) -> pd.DataFrame:
        out = self._merge(X=X, keep_cols=keep_cols)
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

    def inverse_transform(self, X: pd.DataFrame, keep_cols: tuple = (), **kwargs) -> pd.DataFrame:
        out = self._merge(X=X, keep_cols=keep_cols)
        for value_colname in self.value_colnames:
            if self.scale:
                out[value_colname] *= out[f"{value_colname}__std"]
            out.pop(f"{value_colname}__std")
            if self.center:
                out[value_colname] += out[f"{value_colname}__mean"]
            out.pop(f"{value_colname}__mean")

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
    df_diffs = pd.concat(
        [pd.DataFrame({'_to_add': np.arange(0, dt_diff + end_inclusive) * td}).assign(_dt_diff=dt_diff * td)
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



