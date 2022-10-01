from torch import nn
import torch.nn.functional as F
import torch
from rec4torch.activations import activation_layer


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


class FM(nn.Module):
    """FM因子分解机的实现, 使用二阶项简化来计算交叉部分
    inputs: [btz, field_size, emb_size]
    output: [btz, 1]
    """
    def __init__(self):
        super(FM, self).__init__()
    
    def forward(self, inputs):
        # inputs: [btz, field_size, emb_size]
        square_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)  # [btz, 1, emb_size]
        sum_square = torch.sum(torch.pow(inputs, 2), dim=1, keepdim=True)  # [btz, 1, emb_size]
        return 0.5 * torch.sum(square_sum - sum_square, dim=-1)


class SequencePoolingLayer(nn.Module):
    """seq输入转Pooling，支持多种pooling方式
    """
    def __init__(self, mode='mean', support_masking=False):
        super(SequencePoolingLayer, self).__init__()
        assert mode in {'sum', 'mean', 'max'}, 'parameter mode should in [sum, mean, max]'
        self.mode = mode
        self.support_masking = support_masking
    
    def forward(self, seq_value_len_list):
        # seq_value_len_list: [btz, seq_len, hdsz], [btz, seq_len]/[btz,1]
        seq_input, seq_len = seq_value_len_list

        if self.support_masking:  # 传入的是mask
            mask = seq_len.float()
            user_behavior_len = torch.sum(mask, dim=-1, keepdim=True)  # [btz, 1]
            mask = mask.unsqueeze(2)  # [btz, seq_len, 1]
        else:  # 传入的是behavior长度
            user_behavior_len = seq_len
            mask = torch.arange(0, seq_input.shape[1]) < user_behavior_len.unsqueeze(-1)
            mask = torch.transpose(mask, 1, 2)  # [btz, seq_len, 1]
        
        mask = torch.repeat_interleave(mask, seq_input.shape[-1], dim=2)  # [btz, seq_len, hdsz]
        mask = (1 - mask).bool()
        
        if self.mode == 'max':
            seq_input = torch.masked_fill(seq_input, mask, 1e-8)
            return torch.max(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'sum':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            return torch.sum(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'mean':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            seq_sum = torch.sum(seq_input, dim=1, keepdim=True)
            return seq_sum / (user_behavior_len.unsqueeze(-1) + 1e-8)

class AttentionSequencePoolingLayer(nn.Module):
    """DIN中使用的序列注意力
    """
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization

        # 局部注意力单元
        self.dnn = DNN(inputs_dim=4 * embedding_dim, hidden_units=att_hidden_units, activation=att_activation, 
                       dice_dim=kwargs.get('dice_dim', 3), use_bn=kwargs.get('dice_dim', False), dropout_rate=kwargs.get('dropout_rate', 0))
        self.dense = nn.Linear(att_hidden_units[-1], 1)

    def forward(self, query, keys, keys_length, mask=None):
        """
        query: 候选item, [btz, 1, emb_size]
        keys:  历史点击序列, [btz, seq_len, emb_size]
        keys_len: keys的长度, [btz, 1]
        mask: [btz, seq_len]
        """
        btz, seq_len, emb_size = keys.shape

        # 计算注意力分数
        queries = query.expand(-1, seq_len, -1)
        attn_input = torch.cat([queries, keys, queries-keys, queries*keys], dim=-1)  # [btz, seq_len, 4*emb_size]
        attn_output = self.dnn(attn_input)  # [btz, seq_len, hidden_units[-1]]
        attn_score = self.dense(attn_output)  # [btz, seq_len, 1]

        # Mask处理
        if mask is not None:
            keys_mask = mask.unsqueeze(1)  # [btz, 1, seq_len]
        else:
            keys_mask = torch.arange(seq_len, device=keys.device).repeat(btz, 1)  # [btz, seq_len]
            keys_mask = keys_mask < keys_length
            keys_mask = keys_mask.unsqueeze(1)  # [btz, 1, seq_len]

        attn_score = attn_score.transpose(1, 2)  # [btz, 1, seq_len]
        if self.weight_normalization:
            # padding置为-inf，这样softmax后就是0
            attn_score = torch.masked_fill(attn_score, keys_mask.bool(), -1e-7)
            attn_score = F.softmax(attn_score, dim=-1)  # [btz, 1, seq_len]
        else:
            # padding置为0
            attn_score = torch.masked_fill(attn_score, keys_mask.bool(), 0)
        
        if not self.return_score:
            return torch.matmul(attn_score, keys)  # [btz, 1, emb_size]
        return attn_score

