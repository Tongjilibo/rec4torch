import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rec4torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names, build_input_array
from rec4torch.models import DIEN
from rec4torch.snippets import seed_everything

batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(1024)


def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 4, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'), maxlen=4),
        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4)]

    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2, 3])
    gender = np.array([0, 1, 0, 1])
    item_id = np.array([1, 2, 3, 2])  # 0 is mask value
    cate_id = np.array([1, 2, 1, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3, 0.2])

    hist_item_id = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])


    feature_dict = {'user': uid, 'gender': gender, 'item_id': item_id, 'cate_id': cate_id,
                    'hist_item_id': hist_item_id, 'hist_cate_id': hist_cate_id,
                    'pay_score': score}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])
        feature_columns += [
        VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'), maxlen=4),
        VarLenSparseFeat(SparseFeat('neg_hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'),maxlen=4)]

    data = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    data['target'] = np.array([1, 0, 1, 0])

    train_X, train_y = build_input_array(data, feature_columns, target='target')
    train_X = torch.tensor(train_X, dtype=torch.float, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)[:, None]

    return train_X, train_y, feature_columns, behavior_feature_list



if __name__ == "__main__":
    train_X, train_y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=True)
    train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True) 

    model = DIEN(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=True)
    model.to(device)
    
    model.compile(
        loss=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam(model.parameters()),
    )

    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])