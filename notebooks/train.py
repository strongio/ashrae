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
from ashrae import DATA_DIR, PROJECT_ROOT, meter_mapping
from ashrae.nn import MultiSeriesStateNN, TimeSeriesStateNN
from ashrae.preprocessing import DataFrameScaler
from ashrae.forecaster import Forecaster

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

# +
METER_TYPE = os.environ.get("METER_TYPE", "chilledwater")

NN_PRETRAIN_NUM_EPOCHS = os.environ.get("NN_PRETRAIN_NUM_EPOCHS", 200 if METER_TYPE=='electricity' else 100 )
NN_PRETRAIN_LR = os.environ.get("NN_PRETRAIN_LR", .002 if METER_TYPE=='electricity' else .001 )

KF_TRAIN_NUM_EPOCHS = os.environ.get("KF_TRAIN_NUM_EPOCHS", 25 if METER_TYPE=='electricity' else 10 )
KF_TRAIN_LR = os.environ.get("KF_TRAIN_LR", .01)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", METER_TYPE)
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    df_train_clean = pd.read_feather(os.path.join(PROJECT_ROOT, "clean-data", "df_train_clean.feather"))
except Exception:
    from clean_data import df_train_clean

from prepare_dataset import (
    season_config, colname_config, primary_uses, holidays, dataloader_factory
)

def loss_plot(df_loss: pd.DataFrame):
    return (
            ggplot(pd.DataFrame(df_loss), aes(x='epoch', y='value')) +
            stat_summary(fun_y=np.mean, geom='line') + facet_wrap("~dataset", scales='free') +
            theme_bw() + theme(figure_size=(10, 4))
    )


# -

# ## Dataset

# +
print(f"Preparing {METER_TYPE} dataset...")

df_mt_trainval = df_train_clean.\
    loc[df_train_clean['meter'] == METER_TYPE,:].\
    loc[:,['building_id', 'timestamp', 'meter_reading', 'meter_reading_clean', 'lower_thresh']].\
    reset_index(drop=True).\
    assign(
        is_drop=lambda df: (df['meter_reading'] < df['lower_thresh']).astype('float'),
        meter_reading_clean_pp=lambda df:  np.log1p(df['meter_reading_clean']) # will be modified in next step
    )

# don't train on buildings (a) if geometric mean is less than ~10, (b) if very few unique values:
train_ids = df_mt_trainval.\
    assign(_log1p_mean = lambda df: df.groupby('building_id')['meter_reading_clean_pp'].transform('mean'),
           _nunique = lambda df: df.groupby('building_id')['meter_reading_clean'].transform('nunique')).\
    query("(_log1p_mean > 2.5) & (_nunique > 50)").\
    loc[:,'building_id'].\
    drop_duplicates().\
    sample(frac=.80).tolist()
val_ids = [id for id in df_mt_trainval['building_id'].drop_duplicates() if id not in train_ids]
print(f"Training {len(train_ids)}, validation {len(val_ids)}")

mt_scaler = DataFrameScaler(
        value_colnames=['meter_reading_clean_pp'],
        group_colname='building_id',
        center=True,
        scale=True
).fit(df_mt_trainval)

df_mt_trainval = mt_scaler.transform(df_mt_trainval, keep_cols='all')

dl_train = dataloader_factory(
    df_readings=df_mt_trainval.loc[df_mt_trainval['building_id'].isin(train_ids)],
    batch_size=50,
    drop_colname=False
)
dl_val = dataloader_factory(
    df_readings=df_mt_trainval.loc[df_mt_trainval['building_id'].isin(val_ids)],
    batch_size=100,
    drop_colname=False
)

print("...finished")
# -

# ## NN Module

print(dataloader_factory.predictors)
pretrain_nn_module = MultiSeriesStateNN(
    num_series=dataloader_factory.df_meta_preds['building_id'].max(),
    num_predictors=len(dataloader_factory.predictors),
    hidden=(50,25,15),
    embed_inputs={
        dataloader_factory.predictors.index('primary_use') : {
            'num_embeddings' : len(primary_uses), 
            'embedding_dim' : 3
        },
        dataloader_factory.predictors.index('holiday') : {
            'num_embeddings' : len(holidays) + 1, 
            'embedding_dim' : 2
        }
    }
)
if METER_TYPE != 'electricity':
    sd = torch.load(os.path.join(MODEL_DIR, "../electricity", "pretrain_nn_module_state_dict.pkl"))
    pretrain_nn_module.load_state_dict(sd)
    print("Will initialize pretrain_nn_module w/electricity weights.")

# ### Pretraining

# +
pretrain_nn_module.optimizer = torch.optim.Adam(pretrain_nn_module.parameters(), lr=NN_PRETRAIN_LR)
pretrain_nn_module.df_loss = []

try:
    pretrain_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/pretrain_nn_module_state_dict.pkl"))
    NN_PRETRAIN_NUM_EPOCHS = 0
except FileNotFoundError:
    print(f"Pre-training NN-module for {NN_PRETRAIN_NUM_EPOCHS} epochs...")
    
for epoch in range(NN_PRETRAIN_NUM_EPOCHS):
    for i, (tb, vb) in enumerate(zip_longest(dl_train, dl_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
            batch = batch.split_measures(slice(1), slice(1, None))
            
            y, X = batch.tensors
            X[torch.isnan(X)] = 0.0
            y = y.squeeze(-1) # batches support multivariate, but we want to squeeze the singleton dim
            with torch.set_grad_enabled(nm == 'train'):
                prediction = pretrain_nn_module(X, series_idx=batch.group_names if nm == 'train' else None)
                loss = torch.mean( (prediction[y == y] - y[y == y]) ** 2 )
                
            if nm == 'train':
                loss.backward()
                pretrain_nn_module.optimizer.step()
                pretrain_nn_module.optimizer.zero_grad()
            pretrain_nn_module.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            print(f"batch {i}, {nm} loss {loss.item():.3f}")
            
    clear_output(wait=True)        
    print(loss_plot(pretrain_nn_module.df_loss) + ggtitle(f"Epoch {epoch}"))
        
    torch.save(pretrain_nn_module.state_dict(), f"{MODEL_DIR}/pretrain_nn_module_state_dict.pkl")
# -

batch = next(iter(dl_train)).split_measures(slice(1), slice(1, None))
_, X = batch.tensors
X[torch.isnan(X)] = 0.0
df_example = batch.tensor_to_dataframe(
    tensor=pretrain_nn_module(X, series_idx=batch.group_names).unsqueeze(-1),
    times=batch.times(),
    group_names=batch.group_names,
    group_colname='building_id',
    time_colname='timestamp',
    measures=['prediction']
).query("building_id == building_id.sample().item()").merge(df_mt_trainval)
print(
    ggplot(df_example.query("timestamp.dt.month > 6"), aes(x='timestamp')) +
    geom_line(aes(y='meter_reading_clean_pp')) +
    geom_line(aes(y='prediction'), color='red', alpha=.60, size=1.5) +
    theme(figure_size=(12,5)) +
    ggtitle(str(df_example['building_id'].unique().item()))
)

# #### Copy Pretraining

# +
pred_nn_module = TimeSeriesStateNN(**pretrain_nn_module._init_kwargs)
for to_param, from_param in zip(pred_nn_module.parameters(), pretrain_nn_module.parameters()):
    to_param.data[:] = from_param.data[:]

# output real-values:
assert isinstance(pred_nn_module.sequential[-1], torch.nn.Tanh)
del pred_nn_module.sequential[-1]

pred_nn_module
# -

# ## Drop Prediction

# +
drop_nn_module = TimeSeriesStateNN(
    **pred_nn_module._init_kwargs, 
    additional_layers=[
        torch.nn.Linear(pred_nn_module.sequential[-1].out_features, 4, bias=False),
        torch.nn.LogSoftmax()
    ]
)

for param_name, from_param in pred_nn_module.named_parameters():
    to_param = dict(drop_nn_module.named_parameters())[param_name]
    to_param.data[:] = from_param.data[:]

drop_nn_module.loss_fun = torch.nn.NLLLoss()
    
drop_nn_module
# -

# TODO: use lower-bound and/or lagged values as predictors
dl_drops_train = dataloader_factory(
    df_readings=df_mt_trainval.loc[df_mt_trainval['building_id'].isin(train_ids)],
    batch_size=50,
    drop_colname='is_drop',
    reading_colname=False
)
dl_drops_val = dataloader_factory(
    df_readings=df_mt_trainval.loc[df_mt_trainval['building_id'].isin(val_ids)],
    batch_size=100,
    drop_colname='is_drop',
    reading_colname=False
)

# +
drop_nn_module.optimizer = torch.optim.Adam(drop_nn_module.parameters(), lr=NN_PRETRAIN_LR)
drop_nn_module.df_loss = []

try:
    drop_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/drop_nn_module_state_dict.pkl"))
    NN_PRETRAIN_NUM_EPOCHS = 0
except FileNotFoundError:
    print(f"Pre-training NN-module for {NN_PRETRAIN_NUM_EPOCHS} epochs...")
    
for epoch in range(NN_PRETRAIN_NUM_EPOCHS):
    for i, (tb, vb) in enumerate(zip_longest(dl_drops_train, dl_drops_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
            batch = batch.split_measures(slice(1), slice(1, None))
            
            y, X = batch.tensors
            X[torch.isnan(X)] = 0.0
            y = y.squeeze(-1) # batches support multivariate, but we want to squeeze the singleton dim
            with torch.set_grad_enabled(nm == 'train'):
                prediction = drop_nn_module(X)
                loss = drop_nn_module.loss_fun(input=prediction[y == y].unsqueeze(-1),
                                               target=y[y == y].unsqueeze(-1).to(torch.long))
                
            if nm == 'train':
                loss.backward()
                drop_nn_module.optimizer.step()
                drop_nn_module.optimizer.zero_grad()
            drop_nn_module.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            print(f"batch {i}, {nm} loss {loss.item():.3f}")
            
    clear_output(wait=True)        
    print(loss_plot(drop_nn_module.df_loss) + ggtitle(f"Epoch {epoch}"))
        
    torch.save(drop_nn_module.state_dict(), f"{MODEL_DIR}/drop_nn_module_state_dict.pkl")
# -

# ## KF

# +
kf = Forecaster(
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
            input_dim=len(dataloader_factory.predictors), 
            state_dim=pred_nn_module.sequential[-1].out_features,
            nn_module=pred_nn_module,
            add_module_params_to_process=False
        ).add_measure('meter_reading_clean_pp')
    ],
     measures=['meter_reading_clean_pp']
    )

# better init:
if METER_TYPE == 'electricity':
    kf.design.process_covariance.set(kf.design.process_covariance.create().data / 100.)
else:
    pred_nn_module.load_state_dict(torch.load(os.path.join(MODEL_DIR, "../electricity", "pred_nn_module_state_dict.pkl")))
    kf.load_state_dict(torch.load(os.path.join(MODEL_DIR, "../electricity", "kf_state_dict.pkl")))
    print("Will initialize KF w/electricity weights.")

# optimizer:
kf.optimizer = torch.optim.Adam(kf.parameters(), lr=KF_TRAIN_LR)
kf.optimizer.add_param_group({'params' : pred_nn_module.parameters(), 'lr' : KF_TRAIN_LR / 4})

# +
try:
    kf.load_state_dict(torch.load(f"{MODEL_DIR}/kf_state_dict.pkl"))
    pred_nn_module.load_state_dict(torch.load(f"{MODEL_DIR}/pred_nn_module_state_dict.pkl"))
    KF_TRAIN_NUM_EPOCHS = 0
except FileNotFoundError:
    print(f"Training KF for {KF_TRAIN_NUM_EPOCHS} epochs...")

kf.df_loss = []
for epoch in range(KF_TRAIN_NUM_EPOCHS):
    for i, (tb, vb) in enumerate(zip_longest(dl_train, dl_val)):
        for nm, batch in {'train' : tb, 'val' : vb}.items():
            if not batch:
                continue
            batch = batch.split_measures(slice(1), slice(1, None))
            batch.tensors[1][torch.isnan(batch.tensors[1])] = 0.0 
            with torch.set_grad_enabled(nm == 'train'):
                loss = kf.forward_backward(
                    batch=batch, 
                    delete_interval='60D', 
                    random_state=rs if nm == 'val' else None
                )
            if nm == 'train':
                kf.optimizer.zero_grad()
            kf.df_loss.append({'value' : loss.item(), 'dataset' : nm, 'epoch' : epoch})
            print(f"{nm} batch {i} loss {loss.item():.3f}")
    clear_output(wait=True)
    print(loss_plot(kf.df_loss) + ggtitle(f"Epoch {epoch}"))
    
    torch.save(kf.state_dict(), f"{MODEL_DIR}/kf_state_dict.pkl")
    torch.save(pred_nn_module.state_dict(), f"{MODEL_DIR}/pred_nn_module_state_dict.pkl")
# -

# ## Validation

# +
val_forecast_dt = np.datetime64('2016-06-01')

df_val_forecast = []
for batch in dl_val:
    batch = batch.split_measures(slice(1), slice(1, None))
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
    df = pred.to_dataframe(batch, **colname_config).\
      query("measure == 'meter_reading_clean_pp'").\
      drop(columns=['measure'])
    df_val_forecast.append(df)
df_val_forecast = pd.concat(df_val_forecast)

# +
df_example = df_val_forecast.\
             query("(building_id == building_id.sample().item()) & (timestamp.dt.month >= 2)").\
             merge(df_mt_trainval, how='left')

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

# +
df_val_err = df_val_forecast.\
              merge(df_mt_trainval, how='left').\
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
    geom_hline(yintercept=0.0) 
)
# -
# ## Export

# +
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_test = pd.read_csv(os.path.join(DATA_DIR,"test.csv"), parse_dates=['timestamp'])
df_test['meter'] = df_test['meter'].map(meter_mapping).astype('category')
df_mt_test = pd.concat([
    df_mt_trainval.\
        loc[:,['building_id','timestamp','meter_reading_clean_pp']].\
        assign(row_id=-1).\
        reset_index(drop=True),
    df_test.\
        loc[df_test['meter']==METER_TYPE,:].\
        drop(columns=['meter']).\
        reset_index(drop=True)
], sort=True)

df_mt_test['chunk'] = np.floor(df_mt_test['building_id'] / 50).astype('int')

for chunk, df_in in df_mt_test.groupby('chunk'):
    dl = dataloader_factory(df_in, reading_colname='meter_reading_clean_pp', batch_size=50)
    assert len(dl) == 1
    batch = next(iter(dl)).split_measures(slice(1), slice(1, None))
    with torch.no_grad():
        readings, predictors = (t.clone() for t in batch.tensors)
        predictors[torch.isnan(predictors)] = 0.
        pred = kf(
            readings,
            start_datetimes=batch.start_datetimes,
            predictors=predictors,
            progress=tqdm_notebook
        )
    df_pred = pred.to_dataframe(batch, **colname_config).\
      query("measure == 'meter_reading_clean_pp'").\
      drop(columns=['measure'])
    
    df_out = df_pred.\
      merge(df_in.loc[:,['building_id', 'timestamp', 'row_id']], on=['building_id', 'timestamp']).\
      rename(columns={'predicted_mean' : 'meter_reading_clean_pp'}).\
      pipe(mt_scaler.inverse_transform, keep_cols=['row_id']).\
      assign(meter_reading = lambda df: np.expm1(df['meter_reading_clean_pp'])).\
      query("row_id >= 0").\
      loc[:,['row_id', 'meter_reading']].\
      reset_index(drop=True)
    path = os.path.join(OUTPUT_DIR, f"{METER_TYPE}-{chunk}.feather")
    print(path)
    df_out.to_feather(path)
# -


