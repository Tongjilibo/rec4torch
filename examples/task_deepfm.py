import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from rec4torch.inputs import SparseFeat, VarLenSparseFeat, build_input_tensor
from rec4torch.models import DeepFM
from rec4torch.snippets import sequence_padding

batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data():
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data = pd.read_csv("./datasets/movielens_sample.txt")
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", ]

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_list = sequence_padding(genres_list)
    data['genres'] = genres_list.tolist()

    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                                for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=genres_list.shape[-1], pooling='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    
    train_X, train_y = build_input_tensor(data, linear_feature_columns+dnn_feature_columns, target=['rating'])
    return train_X, train_y, linear_feature_columns, dnn_feature_columns

# 加载数据集
train_X, train_y, linear_feature_columns, dnn_feature_columns = get_data()
train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True) 

# 4.Define Model,compile and train
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=['accuracy']
)


if __name__ == "__main__":
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])
