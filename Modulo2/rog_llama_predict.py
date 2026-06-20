import json
import re
import torch
import argparse
import time
import numpy as np

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from transformers.utils import logging

logging.set_verbosity_error()

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

    lines = [
        x.strip()
        for x in text.split("\n")
        if len(x.strip()) > 0
    ]

    if len(lines) > 1:
        return lines

    answers = re.split(
        r";|,",
        text
    )

    return [
        a.strip()
        for a in answers
        if len(a.strip()) > 0
    ]

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

precision_sum = 0.0
recall_sum = 0.0
f1_sum = 0.0

inference_times = []

with open(
    args.json_path,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)

hits = 0

results = []

for sample in tqdm(dataset):

    start_time = time.time()

    predicted_answers, raw_response = predict(
        sample
    )

    elapsed = time.time() - start_time

    inference_times.append(
        elapsed
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

    predicted_set = set(
        predicted_answers
    )

    intersection = (
        predicted_set &
        gold_answers
    )

    if first_prediction in gold_answers:
        hits += 1

    if len(predicted_set) > 0:

        precision = (
            len(intersection)
            / len(predicted_set)
        )

    else:

        precision = 0.0

    if len(gold_answers) > 0:

        recall = (
            len(intersection)
            / len(gold_answers)
        )

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    precision_sum += precision
    recall_sum += recall
    f1_sum += f1

    correct = (
        first_prediction
        in gold_answers
    )

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

precision_avg = (
    precision_sum
    / len(dataset)
)

recall_avg = (
    recall_sum
    / len(dataset)
)

f1_avg = (
    f1_sum
    / len(dataset)
)

avg_time = np.mean(
    inference_times
)

min_time = np.min(
    inference_times
)

max_time = np.max(
    inference_times
)

std_time = np.std(
    inference_times
)

total_time = np.sum(
    inference_times
)

print()
print("=" * 60)
print(f"Hits@1: {hits1:.4f}")
print(f"Precision: {precision_avg:.4f}")
print(f"Recall: {recall_avg:.4f}")
print(f"F1: {f1_avg:.4f}")
print("=" * 60)

print(f"Tempo total: {total_time:.2f} s")
print(f"Tempo médio: {avg_time:.4f} s")
print(f"Tempo mínimo: {min_time:.4f} s")
print(f"Tempo máximo: {max_time:.4f} s")
print(f"Desvio padrão: {std_time:.4f} s")

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