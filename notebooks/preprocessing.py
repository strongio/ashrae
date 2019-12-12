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
from plotnine import *

import pandas as pd
import numpy as np
# -

# ## Meter Types

df_train = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
meter_mapping = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
df_train['meter'] = df_train['meter'].map(meter_mapping).astype('category')
df_train

df_train.\
    drop_duplicates(['meter','building_id']).\
    assign(meter = lambda df: df['meter'].astype('str') + ";").\
    groupby('building_id')['meter'].sum().\
    reset_index().\
    groupby('meter').\
    count().\
    reset_index().\
    sort_values('building_id', ascending=False)

# ### Spot-Check Examples
#
# **stray observations:**
# - lots of spikes to zero, or near zero
# - long runs of near zero
# - some runs of no std-dev
# - very strange alternating hour-to-hour pattern for steam (eg **925**)
#
# **buildings to come back to:**
# - **287** hotwater is interesting, i'd just remove it in a real project, but if its in the test set unclear what the best approach is
#
# **key takeaways:**
# - for some/many, extreme drops to zero are predictable and part of the pattern -- not just anomalous data that should be removed (e.g. **1262** hotwater)
#   - this applies less for electric, but even for electric it occasionally does. see **835**
#   - so may need a model that predicts anomalies
# - there's some switching behavior even in electric that's going to be really challenging to capture w/o something like IMM. see **407**

df_example = df_train.query("building_id == building_id.sample().item()")
print(
    ggplot(df_example,#.query("timestamp.dt.month == timestamp.dt.month.sample().item()"),
       aes(x='timestamp', y='np.log1p(meter_reading)')) +
    geom_line() +
    facet_wrap("~meter", scales='free_y',ncol=1) +
    ggtitle(str(df_example['building_id'].unique().item())) +
    theme(figure_size=(12,3.5 * df_example['meter'].nunique()))
)

# ## Electricity Data
#
# Electricity is a good one to start with because spikes can be preprocessed out, and generally don't seem to be meaningful.

# +
df_train_electric = df_train.\
    query("meter == 'electricity'").\
    drop(columns=['meter']).\
    sort_values(['timestamp']).\
    reset_index(drop=True).\
    assign(
        near_zero = lambda df: df['meter_reading'] < 0.01,
        building_row_id = lambda df: df.groupby('building_id').cumcount() + 1,
        near_zero_cumu = lambda df: df.groupby('building_id')['near_zero'].cumsum() / df['building_row_id']
    )
all_buildings = df_train_electric['building_id'].unique()

# buildings that are frequently near zero, but that are rarely above 1-2, so this isn't anomalous per se
# TODO: use more sophisticated method (e.g. proportion of weeks where median value was < 10)
small_buildings = df_train_electric.\
    assign(low_value = lambda df: df['meter_reading'] < 10).\
    groupby('building_id')\
    ['low_value'].mean().\
    reset_index().\
    query("low_value > 0.50").\
    loc[:,'building_id']

print(f"Number of 'small' buildings excluded: {len(small_buildings)}")

good_buildings = df_train_electric.\
    query("~building_id.isin(@small_buildings)").\
    query("near_zero_cumu < .5").\
    groupby('building_id')\
    ['near_zero'].mean().\
    reset_index().\
    query("near_zero < 0.10").\
    loc[:,'building_id']

nz_buildings = set(all_buildings) - set(good_buildings) - set(small_buildings)

print(f"Number of buildings with too much near-zero-data excluded: {len(nz_buildings)}")

df_train_electric_clean = df_train_electric.\
    query("building_id.isin(@good_buildings)").\
    query("near_zero_cumu < .5").\
    reset_index(drop=True).\
    drop(columns=['near_zero','near_zero_cumu'])


# -

def clean_electricity_values(df: pd.DataFrame, 
                             roll_days: tuple = (3,14), 
                             multis: tuple = (0.66, 1.33),
                             group_colname: str = 'building_id', 
                             value_colname: str = 'meter_reading'):
    df = df.copy()
    
    # running sd:
    df['rolling_std'] = df.groupby(group_colname)[value_colname].\
        rolling(roll_days[0] * 24, center=True).std().reset_index(0, drop=True)
    df[f'{value_colname}_clean'] = df[value_colname]
    df.loc[df['rolling_std'] <= .01, f'{value_colname}_clean'] = np.nan

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


df_example = clean_electricity_values(df_train_electric_clean.query("building_id == 988")) #building_id.sample().item()
print(
    ggplot(df_example,
       aes(x='timestamp')) +
    geom_line(aes(y='meter_reading'), color='red') +
    geom_line(aes(y='meter_reading_clean')) +
    ggtitle(str(df_example['building_id'].unique().item())) +
    theme(figure_size=(12,3.5)) +
    geom_ribbon(aes(ymin='lower_thresh', ymax='upper_thresh'), fill=None, linetype='dashed', color='black') +
    scale_y_continuous(trans='log1p') 
)

# this takes a little bit
df_train_electric_clean = clean_electricity_values(df_train_electric_clean)
df_train_electric_clean.to_feather("../data/cleaned/df_train_electric_clean.feather")










