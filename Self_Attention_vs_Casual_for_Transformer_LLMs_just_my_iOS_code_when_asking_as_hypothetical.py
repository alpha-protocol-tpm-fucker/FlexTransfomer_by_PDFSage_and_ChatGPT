"""Data: ∀x (x=request+response).

Bidirectional: ∀i∀j Attend(i,j). Train→MLM; Gen→needs sampler; yields embeddings. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

Causal: ∀i∀j (j≤i→Attend(i,j)). Train→NextToken; Gen→sequential; optimal for code completion. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

Goal(code synthesis)→choose Causal.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class JSONDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        self.data = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]['request']
        s = self.data[idx]['response']
        t = self.tokenizer.encode(r + ' ' + s, add_special_tokens=True)[:self.seq_len]
        return torch.tensor(t)

def mask_tokens(x, mask_id, vocab_size, p=0.15):
    m = torch.rand(x.shape) < p
    y = x.clone()
    x[m] = mask_id
    y[~m] = -100
    return x, y

class BidirectionalTransformer(nn.Module):
    """∀i∀j Attend(i,j)"""
    def __init__(self, dim, heads, layers, vocab_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(512, dim)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads), layers)
        self.out = nn.Linear(dim, vocab_size)
    def forward(self, x):
        n = x.size(1)
        p = torch.arange(n, device=x.device)
        h = self.token(x) + self.pos(p)
        h = self.enc(h.transpose(0, 1)).transpose(0, 1)
        return self.out(h)

class CausalTransformer(nn.Module):
    """∀i∀j (j≤i→Attend(i,j))"""
    def __init__(self, dim, heads, layers, vocab_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(512, dim)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim, heads), layers)
        self.out = nn.Linear(dim, vocab_size)
    def forward(self, x):
        n = x.size(1)
        p = torch.arange(n, device=x.device)
        h = self.token(x) + self.pos(p)
        m = nn.Transformer.generate_square_subsequent_mask(n).to(x.device)
        h = self.dec(h.transpose(0, 1), h.transpose(0, 1), tgt_mask=m).transpose(0, 1)
        return self.out(h)

def train_mlm(model, optim, data, mask_id, vocab_size):
    for x in data:
        inp, labels = mask_tokens(x, mask_id, vocab_size)
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        optim.zero_grad(); loss.backward(); optim.step()

def train_ar(model, optim, data):
    for x in data:
        inp = x[:, :-1]
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        optim.zero_grad(); loss.backward(); optim.step()

if __name__ == '__main__':
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = JSONDataset('data.json', tok, 128)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    mask_id = tok.mask_token_id
    vocab = tok.vocab_size
    bi = BidirectionalTransformer(512, 8, 6, vocab)
    ca = CausalTransformer(512, 8, 6, vocab)
    ob = torch.optim.Adam(bi.parameters())
    oc = torch.optim.Adam(ca.parameters())
    train_mlm(bi, ob, dl, mask_id, vocab)
    train_ar(ca, oc, dl)
