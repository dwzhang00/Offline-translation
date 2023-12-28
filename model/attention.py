import torch
import math, copy
import torch.nn as nn

from utils import clones

def attention(query, key, value, scale_num, mask=None, dropout=None):

    # torch.bmm() 仅支持三维输入， 合并nbatches和self.h
    batch_size = query.size(0)
    query = query.reshape(-1, query.size(2), query.size(3))
    key = key.reshape(-1, key.size(2), key.size(3))

    scores = torch.bmm(query, key.permute(0,2,1)) * scale_num

    # 还原四维计算attn
    scores = scores.reshape(batch_size, -1, scores.size(1), scores.size(2))

    if mask is not None:
        scores = scores.masked_fill(mask == False, -1e9)

    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    p_attn = p_attn.reshape(-1, p_attn.size(2), p_attn.size(3))
    value = value.reshape(-1, value.size(2), value.size(3))

    return torch.bmm(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        self.d_k = d_model // h # 每个head的维度, d_model=256, h=8 ,d_k=32
        self.scale_num = 1 / math.sqrt(self.d_k) # scores/d_k 防止sofemax偏移过大
        self.h = h  # head数
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.linears[0](query).reshape(nbatches, -1, self.h, self.d_k).permute(0,2,1,3)
        key = self.linears[1](key).reshape(nbatches, -1, self.h, self.d_k).permute(0,2,1,3)
        value = self.linears[2](value).reshape(nbatches, -1, self.h, self.d_k).permute(0,2,1,3)

        x, self.attn = attention(query, key, value, self.scale_num, mask=mask, dropout=self.dropout)

        x = x.reshape(nbatches, -1, x.size(1), x.size(2))
        x = x.permute(0,2,1,3).reshape(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)