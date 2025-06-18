import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sacrebleu import corpus_bleu
import math
from model import build_transformer, subsequent_mask

# Config
MAX_LEN = 100
BATCH = 32
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
ds = load_dataset("cfilt/iitb-english-hindi")  # cfilt iitb-hi
train = ds["train"].shuffle(seed=42).select(range(20000))
val = ds["validation"]

# Build vocab via SentencePiece
import sentencepiece as spm
with open("corpus.txt","w",encoding='utf-8') as f:
    for ex in train:
        f.write(ex["translation"]["en"]+"\n")
        f.write(ex["translation"]["hi"]+"\n")
spm.SentencePieceTrainer.Train('--input=corpus.txt --model_prefix=sp --vocab_size=32000')
sp = spm.SentencePieceProcessor(); sp.load("sp.model")

SRC_VOCAB = TGT_VOCAB = sp.get_piece_size()

# Dataset
class MT(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        src = [1]+sp.encode(self.data[i]["translation"]["en"], out_type=int)[:MAX_LEN-2]+[2]
        tgt = [1]+sp.encode(self.data[i]["translation"]["hi"], out_type=int)[:MAX_LEN-2]+[2]
        src += [0]*(MAX_LEN-len(src))
        tgt += [0]*(MAX_LEN-len(tgt))
        tgt_input = torch.tensor(tgt[:-1])
        tgt_out = torch.tensor(tgt[1:])
        tgt_out[tgt_out==0] = -100
        
        # Fix mask creation - remove extra dimensions
        src_mask = (torch.tensor(src)!=0)  # Just (L,) shape
        tgt_mask = subsequent_mask(MAX_LEN-1)  # (L, L) shape
        
        return torch.tensor(src), src_mask, tgt_input, tgt_mask, tgt_out

train_dl = DataLoader(MT(train), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(MT(val), batch_size=BATCH)

# Build Model
model = build_transformer(SRC_VOCAB, TGT_VOCAB, MAX_LEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total=0
    for src, src_mask, tgt_in, tgt_mask, tgt_out in train_dl:
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        
        memory = model.encode(src, src_mask)
        dec = model.decode(memory, src_mask, tgt_in, tgt_mask)
        logits = model.generator(dec)
        loss = criterion(logits.view(-1, TGT_VOCAB), tgt_out.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}: loss={(total/len(train_dl)):.4f}")

# Beam Search
def beam_search(src_sentence, k=5, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        src = [1]+sp.encode(src_sentence, out_type=int)[:MAX_LEN-2]+[2]
        src += [0]*(MAX_LEN-len(src))
        src = torch.tensor(src).unsqueeze(0).to(DEVICE)
        src_mask = (src!=0).squeeze(0)  # Remove batch dim for mask
        memory = model.encode(src, src_mask)
        beams = [([1], 0.0)]
        
        for _ in range(max_len-1):
            new_beams = []
            for seq, score in beams:
                if seq[-1]==2:
                    new_beams.append((seq, score))
                    continue
                tgt = torch.tensor(seq).unsqueeze(0).to(DEVICE)
                tgt_mask = subsequent_mask(len(seq)).to(DEVICE)
                dec = model.decode(memory, src_mask, tgt, tgt_mask)
                probs = model.generator(dec)[:,-1:]
                topk = torch.topk(probs, k)
                for i, p in zip(topk.indices[0][0], topk.values[0][0]):
                    new_beams.append((seq+[i.item()], score + p.item()))
            beams = sorted(new_beams, key=lambda x: x[1]/(len(x[0])**0.7), reverse=True)[:k]
            if all(b[0][-1]==2 for b in beams): break
        return beams[0][0]

# Evaluation BLEU
preds, refs = [], []
for ex in val.select(range(500)):
    seq = beam_search(ex["translation"]["en"])
    text = sp.decode_ids(seq[1:-1])
    preds.append(text)
    refs.append(ex["translation"]["hi"])
print("BLEU:", corpus_bleu(preds, [refs]).score)