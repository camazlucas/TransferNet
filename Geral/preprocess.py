import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================================
# CONFIGURAÇÕES
# =========================================

INPUT_DIR = "../../Dataset/WebQSP/TransferNETDataset"
OUTPUT_DIR = "../../Dataset/WebQSP/GeralFormat"

ENTITY_FILE = "entities.dict"
RELATION_FILE = "relations.dict"
TRIPLE_FILE = "FB.txt"

TRAIN_FILE = "qa_train_webqsp.txt"
TEST_FILE = "qa_test_webqsp.txt"

ADD_REVERSE = False

# =========================================
# CRIA PASTA DE SAÍDA
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# CARREGA ENTIDADES
# =========================================

print("Loading entities...")

entities = []

with open(os.path.join(INPUT_DIR, ENTITY_FILE), encoding="utf-8") as f:
    for line in f:
        entity = line.strip().split('\t')[0]
        entities.append(entity)

with open(os.path.join(OUTPUT_DIR, "entities.txt"), "w", encoding="utf-8") as f:
    for e in entities:
        f.write(e + "\n")

print(f"Entities: {len(entities)}")

# =========================================
# CARREGA RELAÇÕES
# =========================================

print("Loading relations...")

relations = []

with open(os.path.join(INPUT_DIR, RELATION_FILE), encoding="utf-8") as f:
    for line in f:
        rel = line.strip().split('\t')[0]
        relations.append(rel)

if ADD_REVERSE:
    reverse_relations = [r + "_reverse" for r in relations]
    relations.extend(reverse_relations)

with open(os.path.join(OUTPUT_DIR, "relations.txt"), "w", encoding="utf-8") as f:
    for r in relations:
        f.write(r + "\n")

print(f"Relations: {len(relations)}")

# =========================================
# CARREGA KG COMPLETO
# =========================================

print("Loading KG triples...")

kg_map = defaultdict(list)

triple_count = 0

with open(os.path.join(INPUT_DIR, TRIPLE_FILE), encoding="utf-8") as f:

    for line in tqdm(f):

        l = line.strip().split('\t')

        if len(l) != 3:
            continue

        h, r, t = l

        kg_map[h].append([h, r, t])

        if ADD_REVERSE:
            kg_map[t].append([t, r + "_reverse", h])

        triple_count += 1

print(f"KG triples: {triple_count}")
print(f"Entities with neighbors: {len(kg_map)}")

# =========================================
# FUNÇÃO DE GERAÇÃO
# =========================================

def convert_split(input_file, output_file):

    results = []

    removed = 0

    print(f"\nProcessing {input_file}...")

    with open(os.path.join(INPUT_DIR, input_file), encoding="utf-8") as f:

        for line in tqdm(f):

            try:
                sample = json.loads(line)

            except:
                removed += 1
                continue

            question = sample.get("question", "").strip()

            topic_entities = sample.get("entities", [])
            answers = sample.get("answers", [])

            if len(topic_entities) == 0:
                removed += 1
                continue

            subgraph = []

            visited = set()

            # =========================================
            # 1-HOP SUBGRAPH
            # =========================================

            for ent in topic_entities:

                if ent not in kg_map:
                    continue

                for triple in kg_map[ent]:

                    triple_tuple = tuple(triple)

                    if triple_tuple not in visited:
                        visited.add(triple_tuple)
                        subgraph.append(triple)

            if len(subgraph) == 0:
                removed += 1
                continue

            output_sample = {
                "question": question,
                "entities": topic_entities,
                "answers": answers,
                "subgraph": {
                    "tuples": subgraph
                }
            }

            results.append(output_sample)

    with open(os.path.join(OUTPUT_DIR, output_file), "w", encoding="utf-8") as f:

        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved: {output_file}")
    print(f"Samples: {len(results)}")
    print(f"Removed: {removed}")

# =========================================
# CONVERSÃO
# =========================================

convert_split(TRAIN_FILE, "train_simple.json")
convert_split(TEST_FILE, "test_simple.json")

print("\nDone.")