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

from utils import DataFrameScaler, TimeSeriesStateNN, loss_plot

import torch
from torch.utils.data import DataLoader

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, FourierSeasonDynamic, Season, NN
from torch_kalman.utils.data import TimeSeriesDataset

torch.manual_seed(2019-12-12)
np.random.seed(2019-12-12)
# -

from prepare_dataset import prepare_dataset, season_config

df_train_clean = pd.read_feather("../data/cleaned/df_train_clean.feather")

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

predictors = list(electric_dataset.measures[1])
pred_nn_module = TimeSeriesStateNN(
    embed_inputs = {
        predictors.index('primary_use') : {
            'num_embeddings' : len(electric_dataset.tensors[1][...,predictors.index('primary_use')].unique()), 
            'embedding_dim' : 3
        },
        predictors.index('hour_in_day') : {'num_embeddings' : 24, 'embedding_dim' : 10},
        predictors.index('holiday') : {
            'num_embeddings' : len(electric_dataset.tensors[1][...,predictors.index('holiday')].unique()), 
            'embedding_dim' : 2
        }
    },
    sub_module = torch.nn.Sequential(
        torch.nn.Linear(len(predictors) - 3 + (3 + 10 + 2), 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 25),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=25, out_features=12, bias=False)
    )
)
print(predictors)

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

df_loss = []
for epoch in range(25):
    for i, batch in enumerate(electric_dl_train):
        kf.optimizer.zero_grad()
        loss = forward_backward(model=kf, batch=batch, delete_interval='7D')
        df_loss.append({'value' : loss.item(), 'dataset' : 'train', 'epoch' : epoch})
        clear_output(wait=True)
        print(loss_plot(df_loss) + ggtitle(f"Epoch {epoch}, train batch {i}"))
        
    for i, batch in enumerate(electric_dl_val):
        with torch.no_grad():
            loss = forward_backward(model=kf, batch=batch, delete_interval='7D', random_state=rs)
        df_loss.append({'value' : loss.item(), 'dataset' : 'val', 'epoch' : epoch})
        clear_output(wait=True)
        print(loss_plot(df_loss) + ggtitle(f"Epoch {epoch}, val batch {i}"))
    
