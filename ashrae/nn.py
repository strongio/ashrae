import torch


class TimeSeriesStateNN(torch.nn.Module):
    def __init__(self,
                 num_predictors: int,
                 hidden: tuple,
                 embed_inputs: dict = None):
        self._init_kwargs = {'num_predictors': num_predictors, 'hidden': hidden, 'embed_inputs': embed_inputs}

        embed_inputs = embed_inputs or {}
        super().__init__()

        # adjust input-dim for embed inputs:
        in_features = num_predictors
        for ei in embed_inputs.values():
            in_features -= 1
            in_features += ei['embedding_dim']

        # create sequential submodule:
        layers = []
        for i, outf in enumerate(hidden):
            layers.append(torch.nn.Linear(in_features=in_features if i == 0 else hidden[i - 1], out_features=outf))
            layers.append(torch.nn.Tanh())
        self.sequential = torch.nn.Sequential(*layers)

        # create embedding modules:
        self.embed_modules = torch.nn.ModuleDict()
        for idx, kwargs in embed_inputs.items():
            self.embed_modules[str(idx)] = torch.nn.Embedding(**kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        forward_idx = list(range(input.shape[-1]))
        to_concat = [None]
        for idx_str, embed_module in self.embed_modules.items():
            idx = int(idx_str)
            to_concat.append(embed_module(input[..., idx].to(torch.long)))
            forward_idx.remove(idx)
        to_concat[0] = input[..., forward_idx]
        return self.sequential(torch.cat(to_concat, dim=-1))

    @classmethod
    def from_multi_series_nn(cls, multi_series_nn: 'MultiSeriesStateNN', num_outputs: int):
        num_series = len(multi_series_nn.biases)

        kwargs = dict(multi_series_nn._init_kwargs)
        kwargs['hidden'] = kwargs['hidden'] + (num_series, num_outputs)

        out = cls(**kwargs)
        # remove output tanh:
        del out.sequential[-1]
        # output doesn't need bias:
        out.sequential[-1].bias.data[:] = 0.
        out.sequential[-1].bias.requires_grad_(False)
        # remove penultimate tanh:
        del out.sequential[-2]
        # copy per-series weights/biases:
        out.sequential[-2].weight.data[:] = torch.stack([w.data for w in multi_series_nn.weights])
        out.sequential[-2].bias.data[:] = torch.cat([b.data for b in multi_series_nn.biases])

        return out


class MultiSeriesStateNN(TimeSeriesStateNN):
    def __init__(self, num_series: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.biases = torch.nn.ParameterList()
        self.weights = torch.nn.ParameterList()
        for layer in reversed(self.sequential):
            try:
                num_weights = layer.out_features
                break
            except AttributeError:
                pass
        for i in range(num_series):
            self.biases.append(torch.nn.Parameter(.001 * torch.randn(1)))
            self.weights.append(torch.nn.Parameter(.001 * torch.randn(num_weights)))

    def forward(self, input: torch.Tensor, series_idx: list = None) -> torch.Tensor:
        prediction = super().forward(input=input)

        if series_idx is None:
            bias = torch.mean(torch.stack(tuple(self.biases)), 0)
            weights = torch.mean(torch.stack(tuple(self.weights)), 0, keepdim=True)
        else:
            bias = []
            weights = []
            for pidx in series_idx:
                bias.append(self.biases[pidx])
                weights.append(self.weights[pidx])
            bias = torch.stack(bias)
            weights = torch.stack(weights)
        return torch.sum(prediction * weights[:, None, :], -1) + bias
