import os
import torch
from torch import nn
from tokenizers import ByteLevelBPETokenizer

# ✅ ---------------- CONFIG ----------------
BLOCK_SIZE = 128
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"✅ Using device: {DEVICE}")

# ✅ ---------------- MODEL ----------------
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

# ✅ ---------------- LOAD TOKENIZER ----------------
tokenizer = ByteLevelBPETokenizer.from_file(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)
vocab_size = tokenizer.get_vocab_size()

# ✅ ---------------- LOAD MODEL ----------------
model = TinyGPT(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE
).to(DEVICE)

# Load your last checkpoint (adjust if needed)
model.load_state_dict(torch.load("tiny_gpt_epoch3.pth", map_location=DEVICE))
model.eval()

print("✅ Loaded model & tokenizer.")

# ✅ ---------------- GENERATE ----------------
def generate(prompt, max_new_tokens=50):
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    model.eval()
    for _ in range(max_new_tokens):
        if input_ids.size(1) >= BLOCK_SIZE:
            input_ids = input_ids[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)

# ✅ ---------------- SAMPLE ----------------
prompt = "The quick brown fox"
output = generate(prompt, max_new_tokens=50)
print(f"\n👉 Prompt: {prompt}\n\n📝 Generated: {output}\n")
