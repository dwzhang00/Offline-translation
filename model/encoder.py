import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from model.sublayer import SublayerConnection
from model.embedding import PositionalEncoding, Embeddings

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout): # size = d_model
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    """
    SublayerConnection的作用就是把multi和ffn连在一起，只不过每一层输出之后都要先norm再残差
    """
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, vocab, d_model, dropout, layer, N):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        # 连续encode 6次，且是循环的encode
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)