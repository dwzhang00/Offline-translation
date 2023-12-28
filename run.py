import copy
import os

from parser import args

import torch
import torch.nn as nn

from prepare_data import PrepareData
from model.attention import MultiHeadedAttention
from model.position_wise_feedforward import PositionwiseFeedForward
from model.transformer import Transformer
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from lib.criterion import LabelSmoothing
from lib.optimizer import NoamOpt
from train import train
from evaluate import evaluate

import linger

def make_model(src_vocab, tgt_vocab, N = 2, d_model = 128, d_ff = 512, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(args.device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(args.device)
    model = Transformer(
        Encoder(src_vocab, d_model, dropout, EncoderLayer(d_model, c(attn), c(ff), dropout).to(args.device), 6).to(args.device),
        Decoder(tgt_vocab, d_model, dropout, DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(args.device), 2).to(args.device),
        ).to(args.device)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(args.device)

def main():
    # 数据预处理
    print("------Start prepare data------")
    data = PrepareData()
    args.src_vocab = len(data.en_word_dict)
    args.tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % args.src_vocab)
    print("tgt_vocab %d" % args.tgt_vocab)
    print("------Finish prepare data------")

    # 初始化模型
    model = make_model(
                        args.src_vocab, 
                        args.tgt_vocab,
                        args.layers, 
                        args.d_model, 
                        args.d_ff,
                        args.h_num,
                        args.dropout
                    )

    # linger量化

    normalize_modules = (nn.Linear, nn.Embedding, nn.LayerNorm)
    model = linger.normalize_layers(model, normalize_modules=normalize_modules, normalize_weight_value=8, normalize_bias_value=None, normalize_output_value=8)
    
    # linger.SetFunctionBmmQuant(True)
    # replace_tuple = (nn.Linear, nn.Embedding, nn.LayerNorm)
    # model = linger.init(model, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
    # model.load_state_dict(torch.load("/data2/user/dwzhang/offline_translation/checkpoints/model_quant_best.pth"), strict=True)

    if args.type == 'train':
        # 训练
        print("------Start train------")
        criterion = LabelSmoothing(args.tgt_vocab, padding_idx = 0, smoothing = 0.0)
        optimizer = NoamOpt(args.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
        train(data, model, criterion, optimizer, False)
        print("------Finish train------")

    elif args.type == "evaluate":
        # 预测
        print("------Start evaluate------")
        # model.load_state_dict(torch.load('/data2/user/dwzhang/offline_translation/checkpoints/model_best.pth'), strict=True)
        evaluate(data, model)
        print("------Finish evaluate------")

    else:
        print("Error: please select type within [train / evaluate]")

if __name__ == "__main__":
    main()