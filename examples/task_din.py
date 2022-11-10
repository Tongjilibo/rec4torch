import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rec4torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names, build_input_array
from rec4torch.models import DIN
from rec4torch.snippets import seed_everything

batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4)]
    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}
    data = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    data['target'] = np.array([1, 0, 1])

    train_X, train_y = build_input_array(data, feature_columns, target='target')
    train_X = torch.tensor(train_X, dtype=torch.float, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)[:, None]

    return train_X, train_y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    train_X, train_y, feature_columns, behavior_feature_list = get_xy_fd()
    train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True) 

    model = DIN(feature_columns, behavior_feature_list, att_weight_normalization=True)
    model.to(device)

    model.compile(
        loss=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adagrad(model.parameters()),
    )

    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])