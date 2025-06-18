import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sacrebleu import corpus_bleu
import math
import time
from model import build_transformer, subsequent_mask

# OPTIMIZED CONFIG FOR FASTER TRAINING
MAX_LEN = 50  # Reduced from 100
BATCH = 16    # Reduced from 32 for CPU
EPOCHS = 2    # Reduced for testing
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Load smaller dataset for testing
ds = load_dataset("cfilt/iitb-english-hindi")
# Use much smaller dataset for testing
train = ds["train"].shuffle(seed=42).select(range(1000))  # Only 1000 samples
val = ds["validation"].select(range(100))  # Only 100 samples for validation

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(val)}")

# Build vocab via SentencePiece (skip if files already exist)
import sentencepiece as spm

if not os.path.exists("sp.model"):
    print("Training SentencePiece model...")
    with open("corpus.txt","w",encoding='utf-8') as f:
        for ex in train:
            f.write(ex["translation"]["en"]+"\n")
            f.write(ex["translation"]["hi"]+"\n")
    
    # Smaller vocab for faster processing
    spm.SentencePieceTrainer.Train('--input=corpus.txt --model_prefix=sp --vocab_size=8000 --character_coverage=0.995')
else:
    print("Using existing SentencePiece model...")

sp = spm.SentencePieceProcessor()
sp.load("sp.model")
SRC_VOCAB = TGT_VOCAB = sp.get_piece_size()
print(f"Vocabulary size: {SRC_VOCAB}")

# Optimized Dataset
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
        
        src_mask = (torch.tensor(src)!=0)
        tgt_mask = subsequent_mask(MAX_LEN-1)
        
        return torch.tensor(src), src_mask, tgt_input, tgt_mask, tgt_out

train_dl = DataLoader(MT(train), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(MT(val), batch_size=BATCH)

# Build smaller model for faster training
print("Building model...")
model = build_transformer(
    SRC_VOCAB, TGT_VOCAB, MAX_LEN,
    d_model=256,  # Reduced from 512
    N=3,          # Reduced from 6 layers
    h=4,          # Reduced from 8 heads
    d_ff=1024,    # Reduced from 2048
    dropout=0.1
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher learning rate
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Quick test function
def quick_test():
    """Test if model can process a single batch without errors"""
    print("\n=== QUICK TEST ===")
    model.eval()
    with torch.no_grad():
        for src, src_mask, tgt_in, tgt_mask, tgt_out in train_dl:
            src, tgt_in = src.to(DEVICE), tgt_in.to(DEVICE)
            src_mask, tgt_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE)
            
            print(f"Input shapes - src: {src.shape}, tgt_in: {tgt_in.shape}")
            print(f"Mask shapes - src_mask: {src_mask.shape}, tgt_mask: {tgt_mask.shape}")
            
            # Test encoding
            memory = model.encode(src, src_mask)
            print(f"Memory shape: {memory.shape}")
            
            # Test decoding
            dec = model.decode(memory, src_mask, tgt_in, tgt_mask)
            print(f"Decoder output shape: {dec.shape}")
            
            # Test generation
            logits = model.generator(dec)
            print(f"Logits shape: {logits.shape}")
            
            print("✅ Model test passed!")
            break
    print("=================\n")

# Run quick test
quick_test()

# Training with progress tracking
def train_model():
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (src, src_mask, tgt_in, tgt_mask, tgt_out) in enumerate(train_dl):
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            src_mask, tgt_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE)
            
            # Forward pass
            memory = model.encode(src, src_mask)
            dec = model.decode(memory, src_mask, tgt_in, tgt_mask)
            logits = model.generator(dec)
            loss = criterion(logits.view(-1, TGT_VOCAB), tgt_out.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_dl)}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")
        
        avg_loss = total_loss / len(train_dl)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

# Simple translation test
def test_translation():
    """Test the model with a simple translation"""
    print("\n=== TRANSLATION TEST ===")
    model.eval()
    
    test_sentence = "Hello world"
    print(f"Input: {test_sentence}")
    
    with torch.no_grad():
        # Encode input
        src = [1] + sp.encode(test_sentence, out_type=int)[:MAX_LEN-2] + [2]
        src += [0] * (MAX_LEN - len(src))
        src = torch.tensor(src).unsqueeze(0).to(DEVICE)
        src_mask = (src != 0).squeeze(0)
        
        # Get memory
        memory = model.encode(src, src_mask)
        
        # Simple greedy decoding
        tgt = [1]  # Start with BOS token
        for _ in range(MAX_LEN - 1):
            tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(DEVICE)
            tgt_mask = subsequent_mask(len(tgt)).to(DEVICE)
            
            dec = model.decode(memory, src_mask, tgt_tensor, tgt_mask)
            prob = model.generator(dec)[:, -1:]
            next_token = prob.argmax(dim=-1).item()
            
            if next_token == 2:  # EOS token
                break
            tgt.append(next_token)
        
        # Decode output
        output = sp.decode_ids(tgt[1:])  # Remove BOS token
        print(f"Output: {output}")
    print("========================\n")

if __name__ == "__main__":
    # Train the model
    train_model()
    
    # Test translation
    test_translation()
    
    print("Training completed! 🎉")