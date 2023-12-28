import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from model.sublayer import SublayerConnection
from model.embedding import PositionalEncoding
from model.generator import Generator

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, dropout, layer, N):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.generator = Generator(d_model, vocab)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return self.generator(x)