import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Folder containing parquet files
folder_path = r"C:\Users\Qihao\llm-from-scratch\OpenWebText"

# Output files
output_train_file = "output_train{}.txt"
output_val_file = "output_val{}.txt"
vocab_file = "vocab.txt"

def parquet_files_in_dir(directory):
    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".parquet"):
                files.append(os.path.join(root, filename))

    return files


# Get parquet files
files = parquet_files_in_dir(folder_path)

total_files = len(files)

split_index = int(total_files * 0.9)

print(f"Found {total_files} parquet files.")


# Vocabulary set
vocab = set()

# Process chunks
current_files = files[:split_index]

with open(output_train_file, "w", encoding="utf-8") as outfile:

    for parquet_file in tqdm(current_files, desc=f"Writing output_train.txt"):

        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)

            # Most OpenWebText parquet files contain a "text" column
            if "text" not in df.columns:
                print(f"Skipping {parquet_file} (no 'text' column)")
                continue

            # Write each text sample
            for text in df["text"]:

                if not isinstance(text, str):
                    continue

                outfile.write(text)
                outfile.write("\n")

                # Update vocab
                vocab.update(set(text))

        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")


current_files = files[split_index:]

with open(output_val_file, "w", encoding="utf-8") as outfile:

    for parquet_file in tqdm(current_files, desc=f"Writing output_val.txt"):

        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)

            if "text" not in df.columns:
                print(f"Skipping {parquet_file} (no 'text' column)")
                continue

            # Write each text sample
            for text in df["text"]:

                if not isinstance(text, str):
                    continue

                outfile.write(text)
                outfile.write("\n")

                # Update vocab
                vocab.update(set(text))

        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")


# Save vocabulary
with open(vocab_file, "w", encoding="utf-8") as vfile:

    for char in sorted(vocab):
        vfile.write(char + "\n")

print("Done.")
print(f"Vocabulary size: {len(vocab)}")