from datasets import load_dataset
import os

def main():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    os.makedirs("data", exist_ok=True)

    output_path = os.path.join("data", "raw_input.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item['text'] + "\n")

    print(f"✅ WikiText sample saved to: {output_path}")

if __name__ == "__main__":
    main()
