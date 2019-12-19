from typing import Sequence

import torch
from ashrae.training import remove_random_dates
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.state_belief import CensoredGaussian

import numpy as np


class Forecaster(KalmanFilter):

    def __init__(self,
                 measures: Sequence[str],
                 processes: Sequence['Process'],
                 probit_measures: Sequence[str] = (),
                 **kwargs):
        super().__init__(measures=measures, processes=processes, **kwargs)
        self.probit_measures = probit_measures
        assert set(probit_measures).issubset(self.design.measures)
        if self.probit_measures:
            self.family = CensoredGaussian

    def forward(self,
                input: torch.Tensor,
                **kwargs) -> 'StateBeliefOverTime':

        obs = []
        lower = []
        upper = []
        for i, measure in enumerate(self.design.measures):
            values = input[..., i]
            if measure in self.probit_measures:
                is_one = values.numpy().astype(bool)
                obs.append(torch.zeros_like(values))
                lower.append(torch.zeros_like(input[..., i]))
                lower[-1][np.where(is_one)] = -np.inf
                upper.append(torch.zeros_like(input[..., i]))
                upper[-1][np.where(~is_one)] = np.inf
            else:
                obs.append(values)
                lower.append(torch.full_like(values, -float('inf')))
                upper.append(torch.full_like(values, float('inf')))

        input = (torch.stack(obs, 2), torch.stack(lower, 2), torch.stack(upper, 2))

        return super().forward(input=input, **kwargs)

    def forward_backward(self,
                         batch: 'TimeSeriesDataset',
                         **kwargs) -> 'TimeSeriesDataset':
        batch = remove_random_dates(batch, which=0, **kwargs)
        readings, predictors = batch.tensors
        prediction = self(readings, start_datetimes=batch.start_datetimes, predictors=predictors)
        lp = prediction.log_prob(readings)
        loss = -lp.mean()
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        return loss
