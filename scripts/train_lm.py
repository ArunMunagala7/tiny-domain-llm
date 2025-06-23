import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer

# ✅ --------------- CONFIG ---------------
BATCH_SIZE = 8
BLOCK_SIZE = 128   # Max sequence length
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
LR = 1e-4
EPOCHS = 3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# ✅ --------------- DATASET ---------------
class TextDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ✅ --------------- MODEL ---------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        h = tok_emb + pos_emb
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return logits

# ✅ --------------- LOAD TOKENIZER ---------------
tokenizer = ByteLevelBPETokenizer.from_file(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# ✅ --------------- LOAD & TOKENIZE CORPUS ---------------
with open("data/clean_corpus.txt", "r") as f:
    text = f.read()

ids = tokenizer.encode(text).ids
print(f"✅ Tokenized corpus: {len(ids)} tokens")

dataset = TextDataset(ids, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ --------------- INIT MODEL ---------------
model = TinyGPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ✅ --------------- TRAINING LOOP ---------------
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{batch_idx+1}/{len(loader)}] Loss: {avg_loss:.4f}")

    print(f"✅ Epoch {epoch} finished. Avg Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f"tiny_gpt_epoch{epoch}.pth")

print("🎉 Training done. Checkpoints saved as tiny_gpt_epochX.pth")
