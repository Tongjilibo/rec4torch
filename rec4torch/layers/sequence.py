import torch
import torch.nn as nn
import torch.nn.functional as F


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