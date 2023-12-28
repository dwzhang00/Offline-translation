import torch
import torch.nn as nn

class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab) # 256*vocab

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)