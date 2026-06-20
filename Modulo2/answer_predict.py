import json
import re
from tqdm import tqdm
from llama_cpp import Llama
import argparse


# ==========================================================
# CONFIG
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    required=True
)

parser.add_argument(
    "--json_path",
    type=str,
    required=True
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True
)

args = parser.parse_args()


# ==========================================================
# LLM
# ==========================================================

llm = Llama(
    model_path=args.model_path,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False
)


# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(sample):

    reasoning_paths = []

    for key in sample:

        if not key.startswith("hop_"):
            continue

        reasoning_paths.append(f"{key}:")

        for triple_data in sample[key]:

            subj, rel, obj = triple_data[0]

            reasoning_paths.append(
                f"{subj} -> {rel} -> {obj}"
            )

        reasoning_paths.append("")

    reasoning_text = "\n".join(reasoning_paths)

    prompt = f"""
Based on the reasoning paths, please answer the given question.

Please keep the answer as simple as possible and return all the possible answers as a list.

Reasoning Paths:
{reasoning_text}

Question:
{sample["question"]}

Return ONLY a JSON in the format:

{{
    "answers": [
        "answer_1",
        "answer_2"
    ]
}}
"""

    return prompt


# ==========================================================
# PARSER
# ==========================================================

def parse_response(text):

    try:

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:

            result = json.loads(
                match.group()
            )

            answers = result.get(
                "answers",
                []
            )

            return [
                str(x).strip()
                for x in answers
            ]

    except Exception:
        pass

    return []


# ==========================================================
# INFERENCE
# ==========================================================

def predict(sample):

    prompt = build_prompt(sample)

    output = llm(
        prompt,
        max_tokens=128,
        temperature=0,
        stop=[
            "</s>"
        ]
    )

    text = output["choices"][0]["text"]

    return parse_response(text)


# ==========================================================
# MAIN
# ==========================================================

with open(
    args.json_path,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)

hits = 0

results = []

for sample in tqdm(dataset):

    predicted_answers = predict(sample)

    gold_answers = set(
        sample["gold_answers"]
    )

    if len(predicted_answers) > 0:

        first_prediction = (
            predicted_answers[0]
            .strip()
        )

    else:

        first_prediction = ""

    correct = (
        first_prediction
        in gold_answers
    )

    if correct:
        hits += 1

    results.append(
        {
            "question": sample["question"],
            "head_entity": sample["head_entity"],
            "prediction": predicted_answers,
            "gold_answers": sample["gold_answers"],
            "correct": correct
        }
    )

hits1 = hits / len(dataset)

print()
print("=" * 60)
print(f"Hits@1: {hits1:.4f}")
print("=" * 60)

with open(
    args.output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    f"Predições salvas em: {args.output_path}"
)