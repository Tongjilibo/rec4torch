from torch import nn
import torch

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