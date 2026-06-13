import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from .data import load_data
from .model import TransferNet

from collections import defaultdict
import json

from IPython import embed


def validate(args, model, data, device, kg_index, verbose = False):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    results = []
    MAX_TAILS_PER_REL = 3
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device)) # [bsz, Esize

            for i in range(len(batch[0])):
                question_ids = batch[1]["input_ids"][i].tolist()

                question = data.tokenizer.decode(
                    question_ids,
                    skip_special_tokens=True
                )

                head_id = batch[0][i].argmax().item()

                current_paths = [
                    {
                        "entity": head_id,
                        "triples": [],
                        "score": 1.0
                    }
                ]

                for hop in range(len(outputs["rel_probs"])):

                    rel_probs = outputs["rel_probs"][hop][i]

                    selected_rel_ids = (
                        rel_probs > 0.8
                    ).nonzero().squeeze(1).tolist()

                    if len(selected_rel_ids) == 0:

                        _, top_rel_ids = torch.topk(
                            rel_probs,
                            k=3
                        )

                        selected_rel_ids = top_rel_ids.tolist()

                    new_paths = []

                    for path in current_paths:

                        for rel_id in selected_rel_ids:

                            tails = kg_index.get(
                                (path["entity"], rel_id),
                                []
                            )

                            tails = tails[:MAX_TAILS_PER_REL]

                            for tail_id in tails:

                                new_path = {
                                    "entity": tail_id,
                                    "triples": path["triples"] + [[
                                        data.id2ent[path["entity"]],
                                        data.id2rel[rel_id],
                                        data.id2ent[tail_id]
                                    ]],
                                    "score": (
                                        path["score"]
                                        * float(rel_probs[rel_id])
                                    )
                                }

                                new_paths.append(new_path)

                    if len(new_paths) > 100:

                        new_paths.sort(
                            key=lambda x: x["score"],
                            reverse=True
                        )

                        new_paths = new_paths[:100]

                    if len(new_paths) == 0:
                        break

                    current_paths = new_paths

                current_paths.sort(
                    key=lambda x: x["score"],
                    reverse=True
                )

                clean_paths = []

                for path in current_paths[:10]:

                    clean_paths.append({
                        "triples": path["triples"],
                        "score": path["score"]
                    })

                results.append({
                    "question": question,
                    "head_entity": data.id2ent[head_id],
                    "top_paths": clean_paths
                })

            ######################################################

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

    print(f"Overall accuracy: {overall_acc:.4f}")

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
            "predicted_paths_3hop.json",
            "w",
            encoding="utf-8"
        ) as f:

        ################################ DEBUG ############################
        # if results:
        #     print(
        #         json.dumps(
        #             results[0],
        #             indent=2,
        #             ensure_ascii=False
        #         )
        #     )

        #     return overall_acc
        ##################################################################

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    return overall_acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    parser.add_argument('--bert_name', default='bert-base-uncased')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, triples, train_loader, val_loader = load_data(
    args.input_dir,
    'bert-base-uncased',
    16
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
