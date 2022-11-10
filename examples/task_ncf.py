import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from rec4torch.inputs import SparseFeat, build_input_array
from rec4torch.models import NeuralCF
from rec4torch.snippets import seed_everything

batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

def get_data():
    data = pd.read_csv("./datasets/movielens_sample.txt")
    sparse_features = ["movie_id", "user_id"]

    # 离散变量编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 离散特征和序列特征处理
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    
    # 生成训练样本
    train_X, train_y = build_input_array(data, dnn_feature_columns, target=['rating'])
    train_X = torch.tensor(train_X, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)
    return train_X, train_y, dnn_feature_columns

# 加载数据集
train_X, train_y, dnn_feature_columns = get_data()
train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True) 

# 模型定义
model = NeuralCF(dnn_feature_columns)
model.to(device)

model.compile(
    loss=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=1e-2),
)

if __name__ == "__main__":
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])
