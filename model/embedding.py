import math
import torch
import torch.nn as nn

from parser import args
import linger

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model): # d_model=256
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) #lut: look up table
        self.d_model = math.sqrt(d_model)
        
    def forward(self, x):
        return self.lut(x) * self.d_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=args.device) # 500*256
        position = torch.arange(0., max_len,  device=args.device).unsqueeze(1) # 500*1
        div_term = torch.exp(torch.arange(0., d_model, 2,  device=args.device) *- (math.log(10000.0) / d_model))  # e^(2i/d_model * -ln10000)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 加1个维度

        self.register_buffer('pe', pe)

    def forward(self, x):
        # float
        # y = self.pe[:, :x.size(1)]
        # x = x + y
        # quant
        y = linger.quant_tensor(self, self.pe, name="pos_y_input", mode=linger.QuantMode.QValue)
        x = x + y[:,:x.size(1)]
        return self.dropout(x)
