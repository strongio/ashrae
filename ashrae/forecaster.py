from typing import Sequence

import torch
from ashrae.training import remove_random_dates
from torch_kalman.covariance import PartialCovarianceFromLogCholesky
from torch_kalman.design import Design
from torch_kalman.kalman_filter import KalmanFilter
from torch_kalman.state_belief import CensoredGaussian

import numpy as np


class DesignWithProbit(Design):
    def __init__(self, processes: Sequence['Process'], measures: Sequence[str], probit_measures: Sequence[str]):
        super().__init__(processes=processes, measures=measures)

        self.probit_measures = probit_measures
        assert set(probit_measures).issubset(self.measures)

        # we fix the measure covariance for probit dimensions, since it's not identifiable:
        self.measure_covariance = PartialCovarianceFromLogCholesky(
            full_dim_names=self.measures,
            partial_dim_names=[m for m in self.measures if m not in self.probit_measures],
            diag=1.0
        )


class Forecaster(KalmanFilter):
    design_cls = DesignWithProbit

    def __init__(self,
                 measures: Sequence[str],
                 processes: Sequence['Process'],
                 probit_measures: Sequence[str] = (),
                 **kwargs):
        if probit_measures:
            self.family = CensoredGaussian
        super().__init__(measures=measures, processes=processes, probit_measures=probit_measures, **kwargs)

    def forward(self,
                input: torch.Tensor,
                **kwargs) -> 'StateBeliefOverTime':

        if issubclass(self.family, CensoredGaussian):
            input = self._convert_input_for_censored(input)

        return super().forward(input=input, **kwargs)

    def _convert_input_for_censored(self, input: torch.Tensor) -> tuple:
        obs = []
        lower = []
        upper = []
        for i, measure in enumerate(self.design.measures):
            values = input[..., i]
            if measure in self.design.probit_measures:
                # all zeros:
                obs.append(torch.zeros_like(values))

                is_nan = np.where(np.isnan(values))

                # if one, then uncensored on bottom:
                is_one = values.numpy().astype(bool)
                is_one[is_nan] = True
                lower.append(torch.zeros_like(input[..., i]))
                lower[-1][np.where(is_one)] = -float('inf')

                # if zero, then uncensored on top:
                is_zero = ~values.numpy().astype(bool)
                is_zero[is_nan] = True
                upper.append(torch.zeros_like(input[..., i]))
                upper[-1][np.where(is_zero)] = float('inf')
            else:
                obs.append(values)
                lower.append(torch.full_like(values, -float('inf')))
                upper.append(torch.full_like(values, float('inf')))

        return torch.stack(obs, 2), torch.stack(lower, 2), torch.stack(upper, 2)

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
