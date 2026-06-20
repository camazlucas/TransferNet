import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from .data import load_data
from .model import TransferNet
import time
import numpy as np

from collections import defaultdict
import json

from IPython import embed

from statistics import median, mode



def validate(args, model, data, device, kg_index, verbose = False):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)

    candidate_hits = 0
    candidate_recall_sum = 0.0
    candidate_f1_sum = 0.0

    question_times = []

    total_start = time.time()

    total_triples = 0
    candidate_entity_count = 0
    candidate_entity_count_max = 0
    candidate_sizes = []

    results = []

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device)) # [bsz, Esize]
            for i in range(len(batch[0])):

                q_start = time.time()

                question_ids = batch[1]["input_ids"][i].tolist()

                question = data.tokenizer.decode(
                    question_ids,
                    skip_special_tokens=True
                )

                head_id = batch[0][i].argmax().item()

                gold_answers_n = [
                    data.id2ent[x]
                    for x in batch[2][i]
                        .gt(0.9)
                        .nonzero()
                        .squeeze(1)
                        .tolist()
                ]

                if args.fixed_hops is not None:
                    max_hops = args.fixed_hops
                else:
                    max_hops = (
                        outputs["hop_attn"][i]
                        .argmax()
                        .item()
                        + 1
                    )
                
                question_result = {
                    "question": question,
                    "head_entity": data.id2ent[head_id],
                    "gold_answers": gold_answers_n
                }

                for h in range(max_hops):
                    question_result[f"hop_{h+1}"] = []

                previous_entities = {head_id: 1.0}

                for hop in range(max_hops):

                    rel_probs = outputs["rel_probs"][hop][i]
                    ent_probs = outputs["ent_probs"][hop][i]

                    if args.relation_mode == "softmax":

                        _, top_rel_ids = torch.topk(
                            rel_probs,
                            k=3
                        )

                        selected_rel_ids = top_rel_ids.tolist()

                    else:

                        selected_rel_ids = (
                            rel_probs > 0.7
                        ).nonzero().squeeze(1).tolist()

                        if len(selected_rel_ids) == 0:

                            _, top_rel_ids = torch.topk(
                                rel_probs,
                                k=3
                            )

                            selected_rel_ids = top_rel_ids.tolist()

                    next_entities = {}

                    hop_triples = []

                    for subj, path_score in previous_entities.items():

                        for rel_id in selected_rel_ids:

                            tails = kg_index.get(
                                (subj, rel_id),
                                []
                            )

                            for tail_id in tails:

                                if hop > 0 and tail_id == head_id:
                                    continue

                                entity_score = float(
                                    ent_probs[tail_id]
                                )

                                if entity_score < 1e-6:
                                    continue

                                relation_score = float(
                                    rel_probs[rel_id]
                                )

                                triple_score = (
                                    relation_score
                                    * entity_score
                                )

                                hop_triples.append(
                                    {
                                        "tail_id": tail_id,
                                        "data": [
                                            [
                                                data.id2ent[subj],
                                                data.id2rel[rel_id],
                                                data.id2ent[tail_id]
                                            ],
                                            float(triple_score)
                                        ]
                                    }
                                )

                                new_score = (
                                    path_score
                                    * triple_score
                                )

                                next_entities[tail_id] = max(
                                    next_entities.get(
                                        tail_id,
                                        0.0
                                    ),
                                    new_score
                                )

                                total_triples += 1


                    if len(next_entities) == 0:
                        break

                    sorted_entities = sorted(
                        next_entities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    previous_entities = dict(
                        sorted_entities[:50]
                    )

                    surviving_entities = set(
                        previous_entities.keys()
                    )

                    hop_triples = [
                        t["data"]
                        for t in hop_triples
                        if t["tail_id"] in surviving_entities
                    ]

                    question_result[f"hop_{hop+1}"] = hop_triples


                candidate_answers = set(previous_entities)

                total_hop_triples = (
                    len(question_result.get("hop_1", []))
                    + len(question_result.get("hop_2", []))
                    + len(question_result.get("hop_3", []))
                )

                if total_hop_triples > 50:

                    hop_limits = {
                        "hop_1": 10,
                        "hop_2": 15,
                        "hop_3": 25
                    }

                    for hop_name, limit in hop_limits.items():

                        if hop_name in question_result:

                            question_result[hop_name] = (
                                question_result[hop_name][:limit]
                            )

                results.append(question_result)

                candidate_entity_count += len(candidate_answers)

                candidate_entity_count_max = max(
                    candidate_entity_count_max,
                    len(candidate_answers)
                )

                candidate_sizes.append(
                    len(candidate_answers)
                )

                gold_answers = set(
                    batch[2][i]
                    .gt(0.9)
                    .nonzero()
                    .squeeze(1)
                    .tolist()
                )

                intersection = (
                    candidate_answers &
                    gold_answers
                )

                if len(intersection) > 0:
                    candidate_hits += 1

                if len(gold_answers) > 0:
                    recall = (
                        len(intersection)
                        / len(gold_answers)
                    )
                else:
                    recall = 0.0

                candidate_recall_sum += recall

                if len(candidate_answers) > 0:
                    precision = (
                        len(intersection)
                        / len(candidate_answers)
                    )
                else:
                    precision = 0.0

                if precision + recall > 0:
                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                    )
                else:
                    f1 = 0.0

                candidate_f1_sum += f1

                question_times.append(
                    time.time() - q_start
                )
            
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]

            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)
            for i in range(len(match_score)):
                h = outputs['hop_attn'][i].argmax().item()
                hop_count[h].append(match_score[i])

            if verbose:
                answers = batch[2]
                for i in range(len(match_score)):
                    if match_score[i] == 0:
                        print('================================================================')
                        question_ids = batch[1]['input_ids'][i].tolist()
                        question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        print(' '.join(question_tokens))
                        topic_id = batch[0][i].argmax(0).item()
                        print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        for t in range(model.num_steps):
                            print('>>>>>>> step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x,y in 
                                zip(question_tokens, outputs['word_attns'][t][i].tolist())])
                            print('> Attention: ' + tmp)
                            print('> Relation:')
                            rel_idx = outputs['rel_probs'][t][i].gt(0.9).nonzero().squeeze(1).tolist()
                            for x in rel_idx:
                                print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()))

                            print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        print('----')
                        print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print(' '.join(question_tokens))
                        print(outputs['hop_attn'][i].tolist())
                        embed()

    overall_acc = correct / count

    candidate_hits_rate = (
        candidate_hits / count
    )

    candidate_recall = (
        candidate_recall_sum / count
    )

    candidate_f1 = (
        candidate_f1_sum / count
    )

    total_time = time.time() - total_start


    print(f"Total de triplas recuperadas: {total_triples}")
    print(f"Média por pergunta: {total_triples / count:.2f}")
    print(
        f"Entidades candidatas médias: "
        f"{candidate_entity_count / count:.2f}"
    )
    print(
        f"Máximo de entidades candidatas: "
        f"{candidate_entity_count_max}"
    )

    print(
        f"Mediana candidatos: "
        f"{median(candidate_sizes):.2f}"
    )

    print(
        f"Moda candidatos: "
        f"{mode(candidate_sizes)}"
    )

    print(
        f"P90 candidatos: "
        f"{np.percentile(candidate_sizes, 90):.2f}"
    )

    print(
        f"P95 candidatos: "
        f"{np.percentile(candidate_sizes, 95):.2f}"
    )

    print(f"Hits@1: {overall_acc:.4f}")
    print(f"Candidate Hits: {candidate_hits_rate:.4f}")
    print(f"Candidate Recall: {candidate_recall:.4f}")
    print(f"F1: {candidate_f1:.4f}")
    print(f"Tempo total: {total_time:.2f} s")
    print(f"Tempo medio: {np.mean(question_times):.4f} s")
    print(f"Tempo minimo: {np.min(question_times):.4f} s")
    print(f"Tempo maximo: {np.max(question_times):.4f} s")
    print(f"Desvio padrao: {np.std(question_times):.4f} s")

    hop_report = []

    for hop in range(model.num_steps):

        hop_acc = (
            sum(hop_count[hop]) /
            (len(hop_count[hop]) + 0.1)
        )

        hop_report.append(
            f"{hop+1}-hop: {hop_acc:.4f}"
        )

    print(
        "Pred hop accuracy: "
        + ", ".join(hop_report)
    )

    with open(
        "predicted_paths_transfernet.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    return overall_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    parser.add_argument('--bert_name', default='bert-base-uncased')
    parser.add_argument('--num_steps', type=int, default=3)
    parser.add_argument('--fixed_hops', type=int, default=None)
    parser.add_argument(
        "--relation_mode",
        choices=["softmax", "sigmoid"],
        required=True
    )

    parser.add_argument(
        '--train_file',
        type=str,
        required=True
    )

    parser.add_argument(
        '--test_file',
        type=str,
        required=True
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, triples, train_loader, val_loader = load_data(
    args.input_dir,
    args.bert_name,
    16,
    args.train_file,
    args.test_file
    )
    #################################### Lista de Indices
    # data.id2ent = {v:k for k,v in ent2id.items()}
    # data.id2rel = {v:k for k,v in rel2id.items()}

    kg_index = defaultdict(list)

    for s, p, o in triples.tolist():
        kg_index[(s, p)].append(o)
    #####################################################

    model = TransferNet(args, ent2id, rel2id, triples)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)

    if args.mode == 'vis':
        validate(args, model, val_loader, device, kg_index, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, kg_index, False)

if __name__ == '__main__':
    main()
