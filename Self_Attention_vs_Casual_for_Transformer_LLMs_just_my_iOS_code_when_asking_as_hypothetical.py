import json
import dataclasses
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class JSONDataset(Dataset):
    """∀x (x=request+response)"""
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

def mask_tokens(x, mask_id, p=0.15):
    m = torch.rand(x.shape, device=x.device) < p
    y = x.clone()
    x[m] = mask_id
    y[~m] = -100
    return x, y

@dataclasses.dataclass
class SoftmaxOptions:
    """∀o∈{impl,dtype,temp,drop,k,p}. impl∈{native,fused,flash}. dtype∈{fp32,fp16,bf16}. temp∈ℝ⁺. drop∈[0,1). k∈ℕ. p∈[0,1]."""
    impl: str = 'native'
    dtype: torch.dtype = torch.float16
    temperature: float = 1.0
    dropout: float = 0.0
    top_k: int = 0
    top_p: float = 0.0

def apply_softmax(logits, opts: SoftmaxOptions):
    """∀logits apply chosen o"""
    l = logits.to(opts.dtype) / opts.temperature
    if opts.impl == 'flash' and hasattr(F, 'scaled_softmax'):
        p = F.scaled_softmax(l, scale=1.0, dim=-1)
    elif opts.impl == 'fused' and hasattr(torch.nn.functional, 'softmax'):
        p = torch.nn.functional.softmax(l, dim=-1)
    else:
        p = F.softmax(l, dim=-1)
    if opts.dropout > 0:
        p = F.dropout(p, opts.dropout, training=False)
    if opts.top_k > 0:
        v, _ = torch.topk(p, opts.top_k, dim=-1)
        thresh = v[..., -1, None]
        p = torch.where(p < thresh, torch.zeros_like(p), p)
    if 0.0 < opts.top_p < 1.0:
        s = torch.sort(p, descending=True, dim=-1)
        c = torch.cumsum(s.values, dim=-1)
        m = c - s.values > opts.top_p
        idx = torch.argsort(s.indices, dim=-1)
        mask = torch.gather(m, -1, idx)
        p = torch.where(mask, torch.zeros_like(p), p)
    z = p.sum(dim=-1, keepdim=True)
    return p / z

class BidirectionalTransformer(nn.Module):
    """∀i∀j Attend(i,j)"""
    def __init__(self, dim, heads, layers, vocab_size, softmax_opts: SoftmaxOptions):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(512, dim)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads), layers)
        self.out = nn.Linear(dim, vocab_size)
        self.softmax_opts = softmax_opts
    def forward(self, x):
        n = x.size(1)
        p = torch.arange(n, device=x.device)
        h = self.token(x) + self.pos(p)
        h = self.enc(h.transpose(0, 1)).transpose(0, 1)
        return self.out(h)
    def generate(self, x, steps):
        for _ in range(steps):
            l = self.forward(x)[:, -1]
            p = apply_softmax(l, self.softmax_opts)
            t = torch.multinomial(p, 1)
            x = torch.cat([x, t], dim=1)
        return x

class CausalTransformer(nn.Module):
    """∀i∀j (j≤i→Attend(i,j))"""
    def __init__(self, dim, heads, layers, vocab_size, softmax_opts: SoftmaxOptions):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(512, dim)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim, heads), layers)
        self.out = nn.Linear(dim, vocab_size)
        self.softmax_opts = softmax_opts
    def forward(self, x):
        n = x.size(1)
        p = torch.arange(n, device=x.device)
        h = self.token(x) + self.pos(p)
        m = nn.Transformer.generate_square_subsequent_mask(n).to(x.device)
        h = self.dec(h.transpose(0, 1), h.transpose(0, 1), tgt_mask=m).transpose(0, 1)
        return self.out(h)
    def generate(self, x, steps):
        for _ in range(steps):
            l = self.forward(x)[:, -1]
            p = apply_softmax(l, self.softmax_opts)
            t = torch.multinomial(p, 1)
            x = torch.cat([x, t], dim=1)
        return x

def train_mlm(model, optim, data, mask_id):
    for x in data:
        x = x.to(next(model.parameters()).device)
        inp, labels = mask_tokens(x, mask_id)
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        optim.zero_grad(); loss.backward(); optim.step()

def train_ar(model, optim, data):
    for x in data:
        x = x.to(next(model.parameters()).device)
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
    opts = SoftmaxOptions()
    bi = BidirectionalTransformer(512, 8, 6, vocab, opts).cuda()
    ca = CausalTransformer(512, 8, 6, vocab, opts).cuda()
    ob = torch.optim.Adam(bi.parameters())
    oc = torch.optim.Adam(ca.parameters())
    train_mlm(bi, ob, dl, mask_id)
    train_ar(ca, oc, dl)
    prompt = torch.tensor([tok.cls_token_id]).unsqueeze(0).cuda()
    print(tok.decode(ca.generate(prompt, 20)[0]))
