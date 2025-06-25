import os
import torch
from torch import nn
from tokenizers import ByteLevelBPETokenizer

# Hyperparameters and model config
BLOCK_SIZE = 128
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# Define the same model as in training
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
        tok_emb = self.token_emb(x)                     # (B, T, D)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos)                     # (1, T, D)
        h = tok_emb + pos_emb                           # (B, T, D)
        for layer in self.layers:
            h = layer(h)                                # (B, T, D)
        logits = self.lm_head(h)                        # (B, T, vocab)
        return logits

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    "tokenizer/vocab.json", 
    "tokenizer/merges.txt"
)
vocab_size = tokenizer.get_vocab_size()

# Load model
model = TinyGPT(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(DEVICE)
model.load_state_dict(torch.load("tiny_gpt_epoch10.pth", map_location=DEVICE))
model.eval()

# Generation function using nucleus sampling
def generate(prompt, max_new_tokens=60, top_p=0.9, temperature=0.9):
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)  # (1, T)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > BLOCK_SIZE:
            input_ids = input_ids[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(input_ids)                  # (1, T, vocab)
            next_token_logits = logits[:, -1, :] / temperature  # (1, vocab)

            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Mask tokens beyond top_p
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = 0
            filtered_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            filtered_probs = torch.softmax(filtered_logits, dim=-1)

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, num_samples=1)  # (1, 1)
            next_id = sorted_indices.gather(-1, sampled_index)                # (1, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)

# 🔁 Try a prompt
if __name__ == "__main__":
    prompt = "The doctor diagnosed"
    output = generate(prompt, top_p=0.9, temperature=0.6)
    print(f"\n👉 Prompt: {prompt}\n\n📝 Output: {output}")
