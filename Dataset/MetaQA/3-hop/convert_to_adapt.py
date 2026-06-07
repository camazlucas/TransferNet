from tqdm import tqdm

INPUT_FILE = "valid_3hop.txt"
OUTPUT_FILE = "qa_valid_metaqa_3hop.txt"

with open(INPUT_FILE, encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):

        line = line.strip()

        if not line:
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        question = parts[0]
        answers = parts[1]

        fout.write(f"{question}\t{answers}\n")