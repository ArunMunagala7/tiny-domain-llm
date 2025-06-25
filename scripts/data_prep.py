import os
import re

def clean_text(text: str) -> str:
    """
    Basic cleaning: 
    - Remove extra spaces
    - Remove unwanted special chars
    - Lowercase
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # multiple spaces -> one
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'\"()\-\n ]', '', text)
    return text.strip()

def main():
    # Input: put your raw domain text in data/raw_input.txt
    input_path = os.path.join('data', 'raw_input.txt')
    output_path = os.path.join('data', 'clean_corpus.txt')

    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    cleaned = clean_text(raw)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"✅ Cleaned corpus saved to: {output_path}")

if __name__ == "__main__":
    main()
