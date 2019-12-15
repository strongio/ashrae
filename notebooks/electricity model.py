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
try:
    from plotnine import *
except ImportError:
    from fake_plotnine import *

import numpy as np
import pandas as pd

from copy import copy

from itertools import zip_longest

from utils import DataFrameScaler, TimeSeriesStateNN, loss_plot, forward_backward

import torch
from torch.utils.data import DataLoader

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, FourierSeasonDynamic, Season, NN
from torch_kalman.utils.data import TimeSeriesDataset

from IPython.display import clear_output

torch.manual_seed(2019-12-12)
np.random.seed(2019-12-12)
# -

from prepare_dataset import prepare_dataset, season_config, primary_uses, holidays

df_train_clean = pd.read_feather("../data/cleaned/df_train_clean.feather")

# ## Dataset

# +
print("Preparing electricity dataset...")

df_electric_trainval = df_train_clean.\
    query("meter == 'electricity'").\
    loc[:,['building_id', 'timestamp', 'meter_reading', 'meter_reading_clean']].\
    reset_index(drop=True).\
    assign(meter_reading_clean_pp = lambda df: np.log1p(df['meter_reading_clean']))

electric_scaler = DataFrameScaler(
        value_colnames=['meter_reading_clean_pp'],
        group_colname='building_id',
        center=True,
        scale=True
).fit(df_electric_trainval)

df_electric_trainval = electric_scaler.transform(df_electric_trainval, keep_cols='all')

train_ids = df_electric_trainval['building_id'].drop_duplicates().sample(frac=.75).tolist()
val_ids = [id for id in df_electric_trainval['building_id'].drop_duplicates() if id not in train_ids]
print(f"Training {len(train_ids)}, validation {len(val_ids)}")

electric_dataset = prepare_dataset(df_electric_trainval, 'meter_reading_clean_pp')
electric_dataset.tensors[1][torch.isnan(electric_dataset.tensors[1])] = 0.0
electric_dl_train = DataLoader(electric_dataset.get_groups(train_ids), batch_size=50, collate_fn=TimeSeriesDataset.collate)
electric_dl_val = DataLoader(electric_dataset.get_groups(val_ids), batch_size=50, collate_fn=TimeSeriesDataset.collate)

print("...finished")
# -

# ## NN Module

# +
predictors = list(electric_dataset.measures[1])
pred_nn_module = TimeSeriesStateNN(
    embed_inputs = {
        predictors.index('primary_use') : {
            'num_embeddings' : len(primary_uses), 
            'embedding_dim' : 3
        },
#         predictors.index('hour_in_day') : {
#             'num_embeddings' : 24, 
#             'embedding_dim' : 10
#         },
        predictors.index('holiday') : {
            'num_embeddings' : len(holidays) + 1, 
            'embedding_dim' : 2
        }
    },
    sub_module = torch.nn.Sequential(
        torch.nn.Linear(in_features=len(predictors) - 2 + (3 + 2), out_features=50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 25),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=25, out_features=12, bias=False)
    )
)

print(predictors)
# -

# ## Pretraining NN

pretrain_nn_module = copy(pred_nn_module)
pretrain_nn_module.sub_module = torch.nn.Sequential(
    *pred_nn_module.sub_module[:-1], 
    torch.nn.Linear(in_features=25, out_features=1, bias=True)
)
pretrain_nn_module.optimizer = torch.optim.Adam(pretrain_nn_module.parameters(), lr=.001)

pretrain_nn_module.df_loss = []
for epoch in range(25):
    for i, batches in enumerate(zip_longest(electric_dl_train, electric_dl_val)):
        for is_val, batch in enumerate(batches):
            if not batch:
                continue
            nm = 'val' if is_val else 'train'
            y, X = batch.tensors
            with torch.set_grad_enabled(nm == 'train'):
                prediction = pretrain_nn_module(X)
                loss = torch.mean( (prediction[y==y] - y[y==y]) ** 2 )
            if nm == 'train':
                loss.backward()
                pretrain_nn_module.optimizer.step()
                pretrain_nn_module.optimizer.zero_grad()
            pretrain_nn_module.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            clear_output(wait=True)
            print(loss_plot(pretrain_nn_module.df_loss) + ggtitle(f"Epoch {epoch}, {nm} batch {i}"))

# ## Train KF

# +
kf = KalmanFilter(
    processes=[
        LocalLevel('level').add_measure('meter_reading_clean_pp'),
        LocalLevel('local_level', decay=(.90,.999)).add_measure('meter_reading_clean_pp'),
        FourierSeasonDynamic(
            id='hour_in_day', 
            seasonal_period=24, 
            K=2, 
            decay=(.90,.999), 
            **season_config
        ).add_measure('meter_reading_clean_pp'), 
        NN(
            id='predictors', 
            input_dim=len(predictors), 
            state_dim=pred_nn_module.out_features,
            nn_module=pred_nn_module,
            add_module_params_to_process=False
        ).add_measure('meter_reading_clean_pp')
    ],
     measures=['meter_reading_clean_pp']
    )

# better init:
kf.design.process_covariance.set(kf.design.process_covariance.create().data / 100.)

# optimizer:
kf.optimizer = torch.optim.Adam(kf.parameters(), lr=.01)
kf.optimizer.add_param_group({'params' : pred_nn_module.parameters(), 'lr' : .001})

# +
rs = np.random.RandomState(2019-12-12)

kf.df_loss = []
for epoch in range(25):
    for i, batches in enumerate(zip_longest(electric_dl_train, electric_dl_val)):
        for is_val, batch in enumerate(batches):
            if not batch:
                continue
            nm = 'val' if is_val else 'train'
            with torch.set_grad_enabled(nm == 'train'):
                loss = forward_backward(model=kf, batch=batch, delete_interval='7D')
            if nm == 'train':
                pretrain_nn_module.optimizer.zero_grad()
            kf.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            clear_output(wait=True)
            print(loss_plot(kf.df_loss) + ggtitle(f"Epoch {epoch}, {nm} batch {i}"))
    
        torch.save(kf.state_dict(), "../models/electricity/kf_state_dict.pkl")
        torch.save(pred_nn_module.state_dict(), "../models/electricity/pred_nn_module_state_dict.pkl")
# -

# ## Validation

# +
val_forecast_dt = np.datetime64('2016-06-01')

df_val_forecast = []
for batch in electric_dl_val:
    with torch.no_grad():
        readings, predictors = (t.clone() for t in batch.tensors)
        readings[np.where(batch.times() > val_forecast_dt)] = float('nan')
        predictors[torch.isnan(predictors)] = 0.
        pred = kf(
            readings,
            start_datetimes=batch.start_datetimes,
            predictors=predictors,
            progress=True
        )
    df = pred.to_dataframe({'start_times': batch.start_times, 'group_names': batch.group_names}, **colname_config).\
      query("measure=='meter_reading_clean_pp'").\
      drop(columns=['measure'])
    df_val_forecast.append(df)
df_val_forecast = pd.concat(df_val_forecast)
