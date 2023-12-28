import torch
import torch.nn as nn

from parser import args

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # 输入需要connection的层
        norm = self.norm(x)
        sub = sublayer(norm)
        return x + self.dropout(sub) # 残差结构