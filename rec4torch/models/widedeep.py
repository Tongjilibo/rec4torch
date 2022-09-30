# -*- coding:utf-8 -*-
"""
Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""
import torch
from torch import nn
from .basemodel import RecBase
from ..inputs import combined_dnn_input
from ..layers import DNN

class WideDeep(RecBase):
    """WideDeep的实现
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=1e-4, 
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
        super(WideDeep, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, task=task)

        if len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation, 
                dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        logit = self.linear_model(X)  # [btz, 1]

        if hasattr(self, 'dnn') and hasattr(self, 'dnn_linear'):
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred
