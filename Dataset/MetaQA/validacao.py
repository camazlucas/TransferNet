from tqdm import tqdm

# carrega entidades
ent2id = {}

with open("entities.dict", encoding="utf-8") as f:
    for line in f:
        ent2id[line.strip()] = 1

missing_heads = 0
missing_answers = 0

with open("1-hop/qa_train_metaqa_1hop.txt", encoding="utf-8") as f:

    for line in tqdm(f):

        q, answers = line.strip().split("\t")

        head = q.split("[")[1].split("]")[0].strip()

        if head not in ent2id:
            missing_heads += 1

        for ans in answers.split("|"):

            if ans not in ent2id:
                missing_answers += 1

print("Heads ausentes:", missing_heads)
print("Answers ausentes:", missing_answers)