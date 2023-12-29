import copy
from parser import args
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import PrepareData
from model.attention import MultiHeadedAttention
from model.position_wise_feedforward import PositionwiseFeedForward
from model.transformer import Transformer
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer


# 数据预处理
data = PrepareData()
args.src_vocab = len(data.en_word_dict)
args.tgt_vocab = len(data.cn_word_dict)



def make_model(src_vocab, tgt_vocab, N = 2, d_model = 128, d_ff = 512, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(args.device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(args.device)
    model = Transformer(
        Encoder(src_vocab, d_model, dropout, EncoderLayer(d_model, c(attn), c(ff), dropout).to(args.device), 6).to(args.device),
        Decoder(tgt_vocab, d_model, dropout, DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(args.device), 2).to(args.device),
        ).to(args.device)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(args.device)


# 初始化模型
def transformer():
    model = make_model(
                args.src_vocab, 
                args.tgt_vocab, 
                args.layers, 
                args.d_model, 
                args.d_ff,
                args.h_num,
                args.dropout
                )
    return model