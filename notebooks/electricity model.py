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
from ashrae import DATA_DIR, PROJECT_ROOT
from ashrae.nn import MultiSeriesStateNN, TimeSeriesStateNN
from ashrae.preprocessing import DataFrameScaler
from ashrae.training import forward_backward

try:
    from plotnine import *
    from IPython.display import clear_output
except ImportError:
    from ashrae.fake_plotnine import *
    clear_output = lambda wait: None
    
import os

import numpy as np
import pandas as pd

from itertools import zip_longest

import torch
from torch.utils.data import DataLoader

from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.process import LocalLevel, FourierSeasonDynamic, NN
from torch_kalman.utils.data import TimeSeriesDataset

torch.manual_seed(2019-12-12)
np.random.seed(2019-12-12)
rs = np.random.RandomState(2019-12-12)
# -

from .prepare_dataset import prepare_dataset, season_config, colname_config, primary_uses, holidays


def loss_plot(df_loss: pd.DataFrame):
    return (
            ggplot(pd.DataFrame(df_loss), aes(x='epoch', y='value')) +
            stat_summary(fun_y=np.mean, geom='line') + facet_wrap("~dataset", scales='free') +
            theme_bw() + theme(figure_size=(10, 4))
    )

NUM_EPOCHS_PRETRAIN_NN = os.environ.get("NUM_EPOCHS_PRETRAIN_NN", 200)
NUM_EPOCHS_PRETRAIN_KF = os.environ.get("NUM_EPOCHS_PRETRAIN_KF", 20)
NUM_EPOCHS_TRAIN_KF = os.environ.get("NUM_EPOCHS_TRAIN_KF", 30)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "electricity")
os.makedirs(MODEL_DIR, exist_ok=True)

df_meta = pd.read_csv(os.path.join(DATA_DIR, "building_metadata.csv"))
df_train_clean = pd.read_feather(os.path.join(PROJECT_ROOT, "clean-data", "df_train_clean.feather"))

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

predictors = list(electric_dataset.measures[1])
print(predictors)
pretrain_nn_module = MultiSeriesStateNN(
    num_series=df_meta['building_id'].max(),
    num_predictors=len(predictors),
    hidden=(50,25),
    embed_inputs={
        predictors.index('primary_use') : {
            'num_embeddings' : len(primary_uses), 
            'embedding_dim' : 3
        },
        predictors.index('holiday') : {
            'num_embeddings' : len(holidays) + 1, 
            'embedding_dim' : 2
        }
    }
)

# ### Pretraining

# +
pretrain_nn_module.optimizer = torch.optim.Adam(pretrain_nn_module.parameters(), lr=.002)
pretrain_nn_module.df_loss = []

try:
    pretrain_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/pretrain_nn_module_state_dict.pkl"))
    NUM_EPOCHS_PRETRAIN_NN = 0
except FileNotFoundError:
    print("Pre-training NN-module...")
    
for epoch in range(NUM_EPOCHS_PRETRAIN_NN):
    for i, (tb, vb) in enumerate(zip_longest(electric_dl_train, electric_dl_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
                
            y, X = batch.tensors
            y = y.squeeze(-1) # batches support multivariate, but we want to squeeze the singleton dim
            with torch.set_grad_enabled(nm == 'train'):
                prediction = pretrain_nn_module(X, series_idx=batch.group_names if nm == 'train' else None)
                loss = torch.mean( (prediction[y == y] - y[y == y]) ** 2 )
                
            if nm == 'train':
                loss.backward()
                pretrain_nn_module.optimizer.step()
                pretrain_nn_module.optimizer.zero_grad()
            
            clear_output(wait=True)
            pretrain_nn_module.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            print(loss_plot(pretrain_nn_module.df_loss) + ggtitle(f"Epoch {epoch}, batch {i}, {nm} loss {loss.item():.2f}"))
        
    torch.save(pretrain_nn_module.state_dict(), f"{MODEL_DIR}/pretrain_nn_module_state_dict.pkl")
# -

batch = next(iter(electric_dl_train))
_, X = batch.tensors
df_example = batch.tensor_to_dataframe(
    tensor=pretrain_nn_module(X, series_idx=batch.group_names).unsqueeze(-1),
    times=batch.times(),
    group_names=batch.group_names,
    group_colname='building_id',
    time_colname='timestamp',
    measures=['prediction']
).query("building_id == building_id.sample().item()").merge(df_electric_trainval)
print(
    ggplot(df_example.query("timestamp.dt.month > 6"), aes(x='timestamp')) +
    geom_line(aes(y='meter_reading_clean_pp')) +
    geom_line(aes(y='prediction'), color='red', alpha=.60, size=1.5) +
    theme(figure_size=(12,5)) +
    ggtitle(str(df_example['building_id'].unique().item()))
)

# ## KF

pred_nn_module = TimeSeriesStateNN.from_multi_series_nn(pretrain_nn_module, num_outputs=12)
# freeze the network...
for param in pred_nn_module.parameters():
    param.requires_grad_(False)
# except for the last two layers:
pred_nn_module.sequential[-2].weight.requires_grad_(True)
pred_nn_module.sequential[-2].bias.requires_grad_(True)
pred_nn_module.sequential[-1].weight.requires_grad_(True)
{k:v.requires_grad for k,v in pred_nn_module.named_parameters()}

# ### Pretraining

# +
nn_process = NN(
                id='predictors', 
                input_dim=len(predictors), 
                state_dim=pred_nn_module.sequential[-1].out_features,
                nn_module=pred_nn_module,
                add_module_params_to_process=False
            ).add_measure('meter_reading_clean_pp')

kf_nn_only = KalmanFilter(
     processes=[nn_process],
     measures=['meter_reading_clean_pp']
    )

# better init:
kf_nn_only.design.process_covariance.set(kf_nn_only.design.process_covariance.create().data / 100.)

# optimizer:
kf_nn_only.optimizer = torch.optim.Adam(kf_nn_only.parameters(), lr=.01)
kf_nn_only.optimizer.add_param_group({'params' : pred_nn_module.parameters(), 'lr' : .005})

# +
try:
    kf_nn_only.load_state_dict(torch.load(f"{MODEL_DIR}/pretrain_kf_state_dict.pkl"))
    pred_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/pretrain_pred_nn_module_state_dict.pkl"))
    NUM_EPOCHS_PRETRAIN_KF = 0
except FileNotFoundError:
    print("Pre-training KF NN-process...")

kf_nn_only.df_loss = []
for epoch in range(NUM_EPOCHS_PRETRAIN_KF):
    for i, (tb, vb) in enumerate(zip_longest(electric_dl_train, electric_dl_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
            with torch.set_grad_enabled(nm == 'train'):
                loss = forward_backward(
                    model=kf_nn_only,
                    batch=batch, 
                    delete_interval='60D', 
                    random_state=rs if nm == 'val' else None
                )
            if nm == 'train':
                kf_nn_only.optimizer.zero_grad()
            kf_nn_only.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            clear_output(wait=True)
            print(loss_plot(kf_nn_only.df_loss) + ggtitle(f"Epoch {epoch}, batch {i}, {nm} loss {loss.item():.2f}"))
    
    torch.save(kf_nn_only.state_dict(), f"{MODEL_DIR}/pretrain_kf_state_dict.pkl")
    torch.save(pred_nn_module.state_dict(), f"{MODEL_DIR}/pretrain_pred_nn_module_state_dict.pkl")
# -

# ### Final training

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
        
        nn_process
    ],
     measures=['meter_reading_clean_pp']
    )

# better init:
kf.design.process_covariance.set(kf.design.process_covariance.create().data / 100.)

# optimizer:
kf.optimizer = torch.optim.Adam(kf.parameters(), lr=.02)
kf.optimizer.add_param_group({'params' : pred_nn_module.parameters(), 'lr' : .005})

# +
try:
    kf.load_state_dict(torch.load(f"{MODEL_DIR}/kf_state_dict.pkl"))
    pred_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/pretrain_pred_nn_module_state_dict.pkl"))
    NUM_EPOCHS_TRAIN_KF = 0
except FileNotFoundError:
    print("Training KF...")

kf.df_loss = []
for epoch in range(NUM_EPOCHS_TRAIN_KF):
    for i, (tb, vb) in enumerate(zip_longest(electric_dl_train, electric_dl_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
            with torch.set_grad_enabled(nm == 'train'):
                loss = forward_backward(
                    model=kf, 
                    batch=batch, 
                    delete_interval='60D', 
                    random_state=rs if nm == 'val' else None
                )
            if nm == 'train':
                kf.optimizer.zero_grad()
            kf.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            clear_output(wait=True)
            print(loss_plot(kf.df_loss) + ggtitle(f"Epoch {epoch}, batch {i}, {nm} loss {loss.item():.2f}"))
    
    torch.save(kf.state_dict(), f"{MODEL_DIR}/kf_state_dict.pkl")
    torch.save(pred_nn_module.state_dict(), f"{MODEL_DIR}/pred_nn_module_state_dict.pkl")
# -

# ## Validation

# +
val_forecast_dt = np.datetime64('2016-06-01')

df_val_forecast = []
for batch in electric_dl_val:
    if 1191 not in batch.group_names:
        continue
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

# +
df_example = df_val_forecast.\
            query("(building_id == 1191) & (timestamp.dt.month >= 5)").\
              merge(df_electric_trainval, how='left')

print(
    ggplot(df_example, 
           aes(x='timestamp')) +
    geom_line(aes(y='predicted_mean'), color='red', alpha=.50, size=1) +
    geom_ribbon(aes(ymin='predicted_mean - predicted_std', ymax='predicted_mean + predicted_std'), alpha=.25) +
    geom_line(aes(y='meter_reading_clean_pp')) +
    geom_vline(xintercept=np.datetime64('2016-06-01')) +
    theme(figure_size=(12,5)) +
    ggtitle(str(df_example['building_id'].unique().item()))
)

print(
    ggplot(df_example.assign(train=lambda df: df['timestamp'] < '2016-06-01'), 
           aes(x='timestamp.dt.hour')) +
    stat_summary(aes(y='predicted_mean'), color='red', alpha=.50, size=1) +
    stat_summary(aes(y='meter_reading_clean_pp')) +
    theme(figure_size=(12,5)) +
    facet_wrap("~train") +
    ggtitle(str(df_example['building_id'].unique().item()))
)
# + {}
df_val_err = df_val_forecast.\
              merge(df_electric_trainval, how='left').\
    assign(resid= lambda df: df['predicted_mean'] - df['meter_reading_clean_pp'], # TODO: inverse-transform
           mse = lambda df: df['resid'] ** 2,
           month = lambda df: df['timestamp'].dt.month).\
    groupby(['building_id','month'])\
    ['mse','resid'].mean().\
    reset_index()

print(
    ggplot(df_val_err, aes(x='month')) + 
    stat_summary(aes(y='resid'), fun_data='mean_cl_boot', color='red') +
    stat_summary(aes(y='mse'), fun_data='mean_cl_boot', color='blue') +
    geom_hline(yintercept=(0.0,-.2,-.4)) 
)
# -


