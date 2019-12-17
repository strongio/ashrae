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
from ashrae.preprocessing import clean_readings

try:
    from plotnine import *
except ImportError:
    from ashrae.fake_plotnine import *

import pandas as pd

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
#
# For some building X meters, it just doesn't make sense to try and use a complex model to predict -- their data is degenerate and should be predicted w/heuristics.

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
# useful info for cleaning meters:
df_train = df_train.\
    sort_values(['timestamp']).\
    reset_index(drop=True).\
    assign(
        near_zero = lambda df: df['meter_reading'] < 0.01,
        building_row_id = lambda df: df.groupby('ts_id').cumcount() + 1,
        near_zero_cumu = lambda df: df.groupby('ts_id')['near_zero'].cumsum() / df['building_row_id']
    )

# + {"hideCode": false, "hidePrompt": false}
all_ts_ids = df_train['ts_id'].unique()
print(f"Number of time-series {len(all_ts_ids):,}")

# buildings that are frequently near zero, but that are rarely above 1-2, so this isn't anomalous per se
# TODO: use more sophisticated method (e.g. proportion of weeks where median value was < 10)
excluded_ids = {}
excluded_ids['small'] = df_train.\
    assign(low_value = lambda df: df['meter_reading'] < 10).\
    groupby('ts_id')\
    ['low_value'].mean().\
    reset_index().\
    query("low_value > 0.50").\
    loc[:,'ts_id'].tolist()

print(f"Number of 'small' time-series excluded: {len(excluded_ids['small'])}")

# near zero buildings:
_good = df_train. \
            loc[~df_train['ts_id'].isin(excluded_ids['small']), :]. \
            query("near_zero_cumu < .5"). \
            groupby('ts_id') \
            ['near_zero'].mean(). \
            reset_index(). \
            query("near_zero < 0.10"). \
            loc[:, 'ts_id'].tolist()

excluded_ids['near_zero'] = list(set(all_ts_ids) - set(_good) - set(excluded_ids['small']))

print(f"Number of time-series with too much near-zero-data excluded: {len(excluded_ids['near_zero'])}")

df_train_clean = df_train. \
    loc[~df_train['ts_id'].isin(excluded_ids['small']) & ~df_train['ts_id'].isin(excluded_ids['near_zero']),:].\
    query("near_zero_cumu < .5"). \
    reset_index(drop=True). \
    drop(columns=['near_zero', 'near_zero_cumu'])

# + {"hideCode": false, "hidePrompt": false}
df_example = clean_readings(df_train_clean.query("building_id == building_id.sample().item()")) 
print(
    ggplot(df_example, aes(x='timestamp')) +
    geom_line(aes(y='meter_reading'), color='red') +
     geom_line(aes(y='meter_reading_clean')) +
     ggtitle(str(df_example['building_id'].unique().item())) +
     theme(figure_size=(12,3.5*df_example['ts_id'].nunique())) +
     geom_ribbon(aes(ymin='lower_thresh', ymax='upper_thresh'), fill=None, linetype='dashed', color='black') +
     scale_y_continuous(trans='log1p') +
     facet_wrap("~meter",ncol=1)
)
# -
clean_data_dir = os.path.join(PROJECT_ROOT, "clean-data")
os.makedirs(clean_data_dir, exist_ok=True)

# + {"hideCode": false, "hidePrompt": false}
# this takes a little bit
df_train_clean = clean_readings(df_train_clean, group_colname='ts_id')
df_train_clean. \
    loc[:, ['building_id', 'timestamp', 'meter', 'meter_reading', 'meter_reading_clean']]. \
    to_feather(os.path.join(clean_data_dir, "df_train_clean.feather"))

# + {"hideCode": false, "hidePrompt": false}
# TODO: save small series
# TODO: save degenerate series


