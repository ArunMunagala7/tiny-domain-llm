# 🧠 Tiny Domain-Specific Language Model (TinyGPT)

This project builds a **small transformer-based language model** from scratch using PyTorch — trained on a domain-specific corpus (like biomedical or educational text). It uses:

- Custom tokenization with **Byte-Level BPE**
- A mini-GPT architecture with **TransformerEncoder layers**
- Full training and generation loop
- Nucleus (top-p) sampling for more coherent text generation

---

## 📌 Project Goals

- Train a mini causal language model on **domain-specific text**
- Learn full ML pipeline: tokenization → training → generation
- Use lightweight architecture (can run on CPU or Apple Silicon GPU)

---

## 📂 Folder Structure

```
tiny-domain-llm/
├── data/
│   └── clean_corpus.txt        ← Plain text corpus used for training
├── tokenizer/
│   ├── vocab.json              ← Learned token-to-ID mapping
│   └── merges.txt              ← Merge rules from BPE tokenizer
├── logs/
│   └── train_loss.txt          ← Training loss logged per epoch
├── scripts/
│   ├── train_tokenizer.py      ← Trains the ByteLevelBPE tokenizer
│   ├── train_lm.py             ← Main training loop for the LM
│   └── generate.py             ← Uses trained model to generate text
├── tiny_gpt_epoch10.pth       ← Model checkpoint after 10 epochs
└── README.md                   ← You're here!
```

---

## 🧠 How It Works

### 1. **Tokenizer Training** (`scripts/train_tokenizer.py`)
- Uses HuggingFace's `ByteLevelBPETokenizer`
- Trains on your `clean_corpus.txt`
- Outputs:
  - `vocab.json`: token → ID map
  - `merges.txt`: BPE merge rules

---

### 2. **Language Model Training** (`scripts/train_lm.py`)
- Loads tokenized corpus
- Defines a tiny GPT-like model using:
  - Embeddings
  - Positional encodings
  - 4 TransformerEncoder layers
- Trains via **causal language modeling** (predict next token)
- Saves model checkpoints + loss logs

---

### 3. **Text Generation** (`scripts/generate.py`)
- Loads trained model + tokenizer
- Supports:
  - Top-p (nucleus) sampling
  - Temperature control
- Generates text from any prompt

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repo
```bash
git clone https://github.com/<your-username>/tiny-domain-llm.git
cd tiny-domain-llm
```

### ✅ 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### ✅ 3. Install Dependencies
```bash
pip install torch tokenizers
```

---

## 🚀 Run Instructions

### ✅ Train the Tokenizer
```bash
python scripts/train_tokenizer.py
```

### ✅ Train the Language Model
```bash
python scripts/train_lm.py
```

### ✅ Generate Text
```bash
python scripts/generate.py
```

---

## 🧠 What You'll Learn

- Custom tokenization using BPE
- Building transformer LMs from scratch
- Model training with PyTorch
- Sampling techniques (top-k vs top-p)
- Practical project structuring for LLMs

---

## 🌱 Future Improvements

- Add Gradio app for live demo
- Train on a larger or real domain corpus (e.g., PubMed)
- Add TensorBoard logging
- Convert to HuggingFace-compatible `AutoModel`
- Add model evaluation (perplexity)

---

## 📄 License
MIT License
