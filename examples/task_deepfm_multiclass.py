import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from rec4torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, build_input_array, TensorDataset
from rec4torch.models import DeepFM
from rec4torch.snippets import sequence_padding, seed_everything
from rec4torch.callbacks import Evaluator
from sklearn.metrics import f1_score

batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

def get_data():
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data = pd.read_csv("./datasets/movielens_sample.txt")
    data['rating'] = data['rating'] - 1
    sparse_features = ["movie_id", "user_id", "gender", "age", "zip", ]
    dense_features = ["occupation"]

    # 离散变量编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 序列特征处理
    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_list = sequence_padding(genres_list)
    data['genres'] = genres_list.tolist()

    # 离散特征和序列特征处理
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=genres_list.shape[-1], pooling='mean')]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns + dense_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns + dense_feature_columns
    
    # 生成训练样本
    train_X, train_y = build_input_array(data, linear_feature_columns+dnn_feature_columns, target='rating')
    train_X = torch.tensor(train_X)
    train_y = torch.tensor(train_y)
    return train_X, train_y, linear_feature_columns, dnn_feature_columns

# 加载数据集
train_X, train_y, linear_feature_columns, dnn_feature_columns = get_data()
train_dataloader = DataLoader(TensorDataset(train_X, train_y, device=device), batch_size=batch_size, shuffle=True) 

# 模型定义
model = DeepFM(linear_feature_columns, dnn_feature_columns, out_dim=5)
model.to(device)

model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=1e-2)
)

class MyEvaluator(Evaluator):
    def evaluate(self):
        y_trues, y_preds = [], []
        for X, y in train_dataloader:
            y_prob = self.model.predict(X)
            y_prob = torch.argmax(torch.softmax(y_prob, dim=-1), dim=-1)
            y_trues.append(y.cpu().numpy())
            y_preds.append(y_prob.cpu().numpy())
        
        return {'f1': f1_score(np.concatenate(y_trues), np.concatenate(y_preds), average='macro')}

if __name__ == "__main__":
    evaluator = MyEvaluator(monitor='f1')
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])