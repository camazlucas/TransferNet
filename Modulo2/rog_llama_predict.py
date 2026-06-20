import json
import re
import torch
import argparse

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

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

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=128
)

args = parser.parse_args()

# ==========================================================
# LLM
# ==========================================================

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path
)

print("Carregando modelo...")

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

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

    prompt = f"""Based on the reasoning paths, please answer the given question.

Please keep the answer as simple as possible and return all the possible answers as a list.

Reasoning Paths:
{reasoning_text}

Question:
{sample["question"]}

Answer:
"""

    return prompt

# ==========================================================
# PARSER
# ==========================================================

def parse_response(text):

    text = text.strip()

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

    text = text.replace("\n", " ")

    answers = re.split(
        r";|,",
        text
    )

    answers = [
        a.strip()
        for a in answers
        if len(a.strip()) > 0
    ]

    return answers

# ==========================================================
# INFERENCE
# ==========================================================

def predict(sample):

    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return parse_response(response), response

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

    predicted_answers, raw_response = predict(
        sample
    )

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
            "raw_response": raw_response,
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