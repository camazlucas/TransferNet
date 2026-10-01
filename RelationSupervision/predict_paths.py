import argparse
import csv
import json

import torch
from transformers import AutoTokenizer

from .data import STOP
from .model import TransferNet

# Filtro de candidatos por hop: so relacoes acima desse limiar entram no JSON de
# saida, no maximo TOP_K por hop. Constante (nao argparse) porque e um parametro
# de pos-processamento da predicao, nao do modelo/treino.
REL_PROB_THRESHOLD = 0.15
TOP_K = 3


def predict_paths(model, tokenizer, id2rel, examples, device, max_hops, batch_size=16):
    model.eval()
    results = []

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            batch_questions = [question for question, _ in batch]

            encoded = tokenizer(
                batch_questions,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(encoded)
            rel_probs = outputs['rel_probs']  # list of [bsz, num_relations], one per hop

            for i, (question, gold) in enumerate(batch):
                hops = {}
                for t, probs in enumerate(rel_probs):
                    candidates = [
                        (id2rel[r], probs[i, r].item())
                        for r in range(probs.size(1))
                        if probs[i, r].item() > REL_PROB_THRESHOLD
                    ]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    candidates = candidates[:TOP_K]

                    hops['hop{}'.format(t + 1)] = [
                        {'relation': rel, 'prob': prob} for rel, prob in candidates
                    ]

                results.append({'question': question, 'gold': gold, 'rel': hops})

    return results


def evaluate(results, max_hops):
    # Taxa de "a relacao-gabarito esta entre os candidatos retornados" por hop
    # (nao so o argmax, que e o que o predict.py/validate() ja mede durante o treino).
    hop_hits = [0] * max_hops
    total = len(results)
    full_path_hits = 0

    for r in results:
        all_hops_hit = True
        for t in range(max_hops):
            candidates = {c['relation'] for c in r['rel']['hop{}'.format(t + 1)]}
            hit = r['gold'][t] in candidates
            hop_hits[t] += int(hit)
            all_hops_hit = all_hops_hit and hit
        full_path_hits += int(all_hops_hit)

    per_hop = {
        'hop{}'.format(t + 1): {
            'hits': hop_hits[t],
            'total': total,
            'hit_rate': hop_hits[t] / total if total else 0.0
        }
        for t in range(max_hops)
    }

    return {
        'total_questions': total,
        'per_hop': per_hop,
        'full_path_coverage': full_path_hits / total if total else 0.0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV with "question" and "paths" columns (test set)')
    parser.add_argument('--output', required=True, help='path to write the predicted-paths JSON')
    parser.add_argument('--eval_report', required=True, help='path to write the evaluation report JSON')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--rel2id', required=True, help='path to the rel2id JSON used at training time')
    parser.add_argument('--bert_name', default='bert-base-uncased')
    parser.add_argument('--max_hops', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.rel2id, encoding='utf-8') as f:
        rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, local_files_only=True)

    model = TransferNet(args, rel2id)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)

    examples = []
    with open(args.input, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold = row['paths'].split('|')[:args.max_hops]
            gold += [STOP] * (args.max_hops - len(gold))
            examples.append((row['question'].strip(), gold))

    results = predict_paths(model, tokenizer, id2rel, examples, device, args.max_hops, args.batch_size)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Wrote {} predictions to {}'.format(len(results), args.output))

    report = evaluate(results, args.max_hops)
    with open(args.eval_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Wrote evaluation report to {}'.format(args.eval_report))


if __name__ == '__main__':
    main()
