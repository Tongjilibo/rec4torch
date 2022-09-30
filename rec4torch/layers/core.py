from audioop import bias
from turtle import forward
from torch import nn
import torch
from .activations import activation_layer

class DNN(nn.Module):
    '''MLP的全连接层
    '''
    def __init__(self, input_dim, hidden_units, activation='relu', dropout_rate=0, use_bn=False, init_std=1e-4, dice_dim=3):
        super(DNN, self).__init__()
        assert isinstance(hidden_units, (tuple, list)) and len(hidden_units) > 0, 'hidden_unit support non_empty list/tuple inputs'
        self.dropout = nn.Dropout(dropout_rate)
        hidden_units = [input_dim] + list(hidden_units)

        layers = []
        for i in range(len(hidden_units)-1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))  # 全连接层
            if use_bn:  # BatchNorm
                layers.append(nn.BatchNorm1d(hidden_units[i+1]))
            layers.append(activation_layer(activation, hidden_units[i + 1], dice_dim))
        self.layers = nn.Sequential(*layers)

        for name, tensor in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        # inputs: [btz, ..., input_dim]
        return self.layers(inputs)  # [btz, ..., hidden_units[-1]]


class PredictionLayer(nn.Module):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        super(PredictionLayer, self).__init__()
        assert task in {"binary", "multiclass", "regression"}, "task must be binary,multiclass or regression"
        self.task = task

        if use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))
        
    def forward(self, X):
        output =  X
        if hasattr(self, 'bias'):
            output += self.bias
        if self.task == 'binary':
            output = torch.sigmoid(output)
        return output