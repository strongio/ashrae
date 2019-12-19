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

# + {"hideCode": false, "hidePrompt": false}
from ashrae import DATA_DIR, PROJECT_ROOT

try:
    from plotnine import *
except ImportError:
    from ashrae.fake_plotnine import *

import pandas as pd
import numpy as np

import os

# + {"hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ## Meter Types

# + {"hideCode": false, "hidePrompt": false}
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=['timestamp'])
meter_mapping = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
df_train['meter'] = df_train['meter'].map(meter_mapping).astype('category')
df_train

# + {"hideCode": false, "hidePrompt": false}
df_train.\
    drop_duplicates(['meter','building_id']).\
    assign(meter = lambda df: df['meter'].astype('str') + ";").\
    groupby('building_id')['meter'].sum().\
    reset_index().\
    groupby('meter').\
    count().\
    reset_index().\
    sort_values('building_id', ascending=False)

# + {"hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ### Spot-Check Examples
#
# **stray observations:**
# - lots of spikes to zero, or near zero
# - long runs of near zero
# - some runs of no std-dev
# - very strange alternating hour-to-hour pattern for steam (eg **925**)
#
# **key takeaways:**
# - for some/many, extreme drops to zero are predictable and part of the pattern -- not just anomalous data that should be removed (e.g. **1262** hotwater)
#   - this applies less for electric, but even for electric it occasionally does. see **835**
#   - so may need a model that predicts anomalies
# - there's some switching behavior even in electric that's going to be really challenging to capture w/o something like IMM. see **407**
# - for some (esp. hotwater: eg 287, 171), long runs of 0 w/ no std-dev 
# - notable pattern (at least for chilledwater: eg 171): daily seasons that involve a dip to exactly zero. could handle this with censoring, esp. if you're using tobitFilter anyways.
# - how should long runs of low-but-nonzero-std-dev be handled? ~see 1295 hotwater~ captured as low std-dev

# + {"hideCode": false, "hidePrompt": false}
df_example = df_train.query("building_id == building_id.sample().item()")
print(
    ggplot(df_example.query("timestamp.dt.month == timestamp.dt.month.sample().item()"),
       aes(x='timestamp', y='meter_reading+1')) +
    geom_line() +
    facet_wrap("~meter", scales='free_y',ncol=1) +
    ggtitle(str(df_example['building_id'].unique().item())) +
    theme(figure_size=(12,3.5 * df_example['meter'].nunique())) +
    scale_y_continuous(trans='log10')
)

# + {"hideCode": false, "hidePrompt": false, "cell_type": "markdown"}
# ## Filter Buildings

# + {"hideCode": false, "hidePrompt": false}
# jic cell re-run:
df_train['td_id'] = ''
df_train.pop('td_id')

# add td_id:
df_train = df_train.\
    loc[:,['building_id','meter']].\
    drop_duplicates().\
    assign(ts_id=lambda df: df['building_id'].astype(str) + '_' + df['meter'].astype(str)).\
    merge(df_train)

# + {"hideCode": false, "hidePrompt": false}
df_train = df_train.\
    sort_values(['building_id','timestamp']).\
    reset_index(drop=True)

df_train_clean = df_train.\
    assign(
        date= lambda df: df['timestamp'].dt.date,
        near_zero= lambda df: df['meter_reading'] < 0.001,
        nz_rolling=lambda df: df.groupby('ts_id')['near_zero'].rolling(24 * 7).mean().reset_index(level=0,drop=True)
    ).\
    query("nz_rolling < 0.80").\
    reset_index(drop=True)

excluded_ids = list(set(df_train['ts_id']) - set(df_train_clean['ts_id'])) 

# TODO: '53_electricity' may not be great to exclude
print(
    ggplot(df_train.loc[df_train['ts_id'].isin(excluded_ids),:], aes(x='timestamp')) +
    geom_line(aes(y='meter_reading')) +
     ggtitle("Excluded (will just predict 0 for test)") +
     theme(figure_size=(12,5))  +
     #scale_y_continuous(trans='log1p') +
     facet_wrap("~ts_id", scales='free')
)


# -

# ## Clean Readings

def clean_readings(df: pd.DataFrame,
                   roll_days: tuple = (3, 14),
                   multis: tuple = (0.66, 1.33),
                   group_colname: str = 'ts_id',
                   value_colname: str = 'meter_reading',
                   std_dev_theshold: float = .01):
    df = df.copy()
    df[f'{value_colname}_clean'] = df[value_colname].where(df[value_colname] > 0.0)

    # running sd:
    df['rolling_std'] = df.groupby(group_colname)[value_colname]. \
        rolling(roll_days[0] * 24, center=True).std().reset_index(0, drop=True)
    df.loc[df['rolling_std'] <= std_dev_theshold, f'{value_colname}_clean'] = np.nan

    # running lower bound:
    # two-pass to deal w/tradeoff between avoiding false-alarms on meters like 256,
    # while not letting ones like 1269 through
    # (living with mediocre cleaning for: 746)
    df['_lb'] = df.groupby(group_colname)[f'{value_colname}_clean'].rolling(24, center=True).min().reset_index(0,drop=True)
    p1 = df.groupby(group_colname)['_lb']. \
        rolling(roll_days[1] * 24, center=True).median().reset_index(0, drop=True)
    p2 = df.groupby(group_colname)['_lb']. \
        rolling(roll_days[0] * 24, center=True).median().reset_index(0, drop=True)
    df['_lb_roll'] = (p1 + p2) / 2
    df['_lb_roll'].fillna(df.groupby([group_colname])['_lb_roll'].transform('min'), inplace=True)

    # running upper bound:
    df['_ub'] = df.groupby(group_colname)[f'{value_colname}_clean']. \
        rolling(24, center=True).max().reset_index(0, drop=True)
    df['_ub_roll'] = df.groupby(group_colname)['_ub']. \
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


# + {"hideCode": false, "hidePrompt": false}
df_example = clean_readings(df_train_clean.query("building_id == building_id.sample().item()")) 
print(
    ggplot(df_example, aes(x='timestamp')) +
    geom_line(aes(y='meter_reading'), color='red') +
     geom_line(aes(y='meter_reading_clean')) +
     ggtitle(str(df_example['building_id'].unique().item())) +
     theme(figure_size=(12,3.*df_example['ts_id'].nunique())) +
     geom_ribbon(aes(ymin='lower_thresh', ymax='upper_thresh'), fill=None, linetype='dashed', color='black') +
     scale_y_continuous(trans='log1p') +
     facet_wrap("~meter",ncol=1)
)
# -
# this takes a little bit
df_train_clean = clean_readings(df_train_clean, group_colname='ts_id')

# + {"hideCode": false, "hidePrompt": false}
clean_data_dir = os.path.join(PROJECT_ROOT, "clean-data")
os.makedirs(clean_data_dir, exist_ok=True)
df_train_clean. \
    loc[:, ['building_id', 'timestamp', 'meter', 'meter_reading', 'meter_reading_clean']]. \
    to_feather(os.path.join(clean_data_dir, "df_train_clean.feather"))
# -


