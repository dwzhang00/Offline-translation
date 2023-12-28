import torch
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from parser import args
from utils import seq_padding, subsequent_mask

class PrepareData:
    def __init__(self):

        # 读取数据 并分词
        self.train_en, self.train_cn = self.load_data(args.train_file) # path --> sentence list
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en) # en: sentence list --> dict{word:index}; int:total; words dict{index:word}
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn) # cn: sentence list --> dict{word:index}; int:total; words dict{index:word}

        # id化
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict) # order sentence index list :according to length. words replaced by index
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r') as f:
            for line in tqdm(f, desc="Loading Data", unit=" lines"):
                line = line.strip().split('\t')

                en.append(["BOS"] + line[0].lower()[:-1].split() + ["EOS"])
                cn.append(["BOS"] + [w for w in line[1]] + ["EOS"])

        return en, cn
    
    def build_dict(self, sentences, max_words = 50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        word_dict = dict()
        word_dict['UNK'] = args.UNK # 0
        word_dict['PAD'] = args.PAD # 1
        for index, w in enumerate(ls):
            word_dict[w[0]] = index+2
        index_dict = {v: k for k, v in word_dict.items()}
        
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        length = len(en)

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # sort sentences by English lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=False):
        idx_list = np.arange(0, len(en), batch_size) # start idx of each batch
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]   # a batch of sentences composed of index: 2D matrix
            batch_cn = [cn[index] for index in batch_index]
            batch_en = seq_padding(batch_en)
            batch_cn = seq_padding(batch_cn)
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
