from tqdm import tqdm

KG_FILE = "kb.tsv"

entities = set()
relations = set()

print("Lendo KG...")

with open(KG_FILE, encoding="utf-8") as f:

    for line in tqdm(f):

        line = line.strip()

        if not line:
            continue

        head, rel, tail = line.split("\t")

        entities.add(head)
        entities.add(tail)

        relations.add(rel)

print(f"Entidades encontradas : {len(entities)}")
print(f"Relações encontradas  : {len(relations)}")

# ==========================================
# entities.dict
# ==========================================

entities = sorted(entities)

with open("entities.dict", "w", encoding="utf-8") as f:

    for ent in entities:
        f.write(f"{ent}\n")

print(f"entities.dict salvo com {len(entities)} entidades")

# ==========================================
# relations.dict
# ==========================================

relations = sorted(relations)

with open("relations.dict", "w", encoding="utf-8") as f:

    idx = 0

    for rel in relations:

        f.write(f"{rel}\t{idx}\n")
        idx += 1

        f.write(f"{rel}_reverse\t{idx}\n")
        idx += 1

print(f"relations.dict salvo com {idx} relações")