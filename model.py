import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.size()
        q = self.w_q(query).view(B, L, self.h, self.d_k).transpose(1,2)
        k = self.w_k(key).view(B, -1, self.h, self.d_k).transpose(1,2)
        v = self.w_v(value).view(B, -1, self.h, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Ensure mask is 4D: [B, 1, L_q, L_k]
            if mask.dim() == 1:  # [L_k]
                mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)  # → [1, 1, 1, L_k]
            elif mask.dim() == 2:  # [B, L_k]
                mask = mask.unsqueeze(1).unsqueeze(2)  # → [B, 1, 1, L_k]
            elif mask.dim() == 3:  # [B, L_q, L_k]
                mask = mask.unsqueeze(1)  # → [B, 1, L_q, L_k]
            # Now broadcast to [B, h, L_q, L_k]
            mask = mask.expand(B, self.h, q.size(2), k.size(2))
            
            # Expand to match attention heads
            mask = mask.expand(B, self.h, q.size(2), k.size(2))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1,2).contiguous().view(B, L, -1)
        return self.w_o(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

def subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask == 0

def build_transformer(src_vocab, tgt_vocab, seq_len, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
    attn = MultiHeadAttention(d_model, h, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
    decoder = Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N)
    src_emb = InputEmbeddings(d_model, src_vocab)
    tgt_emb = InputEmbeddings(d_model, tgt_vocab)
    pos_enc_src = PositionalEncoding(d_model, seq_len, dropout)
    pos_enc_tgt = PositionalEncoding(d_model, seq_len, dropout)
    gen = Generator(d_model, tgt_vocab)
    
    model = nn.Module()
    model.encode = lambda src, src_mask: encoder(pos_enc_src(src_emb(src)), src_mask)
    model.decode = lambda memory, src_mask, tgt, tgt_mask: decoder(pos_enc_tgt(tgt_emb(tgt)), memory, src_mask, tgt_mask)
    model.generator = gen
    
    # initialize
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model