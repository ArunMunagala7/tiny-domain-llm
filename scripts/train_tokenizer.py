from tokenizers import ByteLevelBPETokenizer
import os

def main():
    # Paths
    corpus_file = os.path.join('data', 'clean_corpus.txt')
    tokenizer_dir = 'tokenizer'

    # Make sure output dir exists
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Init tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train
    tokenizer.train(files=corpus_file, vocab_size=8000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save
    tokenizer.save_model(tokenizer_dir)
    print(f"✅ Tokenizer saved to: {tokenizer_dir}")

if __name__ == "__main__":
    main()
