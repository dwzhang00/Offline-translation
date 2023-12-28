import torch
import numpy as np

from parser import args
from utils import subsequent_mask
from nltk.translate.bleu_score import corpus_bleu
import datetime

candidates = []
references = []

def log(data, timestamp):

    file = open(f'/data2/user/dwzhang/offline_translation/log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()
# torch.set_printoptions(threshold=float('inf'))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encoder(src, None)
    # print(torch.floor(memory*64+0.5))
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        # print(ys)
        out = model.decoder(ys, memory, None, None)
        out = torch.floor(out*2+0.5)
        # if i==0: print(out[:,-1,:])
        _, next_word = torch.max(out[:,-1,:], dim = 1)
        next_word = next_word.data[0]
        if next_word != 3:  # index[EOS] = 3
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        else:
            break
    return ys

def evaluate(data, model):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.eval()
    with torch.no_grad():
        for i in range(len(data.dev_en)):   # id matrix
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            # log(en_sent, timestamp)

            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            out_cn_sent = "".join(cn_sent)
            candidates.append(list(i for i in cn_sent if i!=' ')[3:-3])
            print(out_cn_sent)
            # log(cn_sent, timestamp)

            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)
            # print(src)
            out = greedy_decode(model, src, src_mask, max_len = args.max_length, start_symbol = data.cn_word_dict["BOS"])
            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                translation.append(sym)
            references.append([translation])
            print("translation: %s" % " ".join(translation))
            # log("translation: " + " ".join(translation) + "\n", timestamp)
        
        print('Corpus BLEU: {:.3f}'.format(corpus_bleu(references, candidates)))