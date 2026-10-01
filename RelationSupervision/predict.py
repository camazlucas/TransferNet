import torch
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device


def validate(args, model, data, device, verbose=False):
    model.eval()
    count = 0
    exact_match = 0
    hop_correct = defaultdict(int)
    hop_total = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, rel_path = batch_device(batch, device)
            outputs = model(questions)
            rel_probs = outputs['rel_probs']  # list of [bsz, num_relations], one per hop

            pred_path = torch.stack([p.argmax(dim=1) for p in rel_probs], dim=1)  # [bsz, max_hops]
            correct_per_hop = pred_path.eq(rel_path)  # [bsz, max_hops]

            count += rel_path.size(0)
            exact_match += correct_per_hop.all(dim=1).sum().item()

            for t in range(rel_path.size(1)):
                hop_correct[t] += correct_per_hop[:, t].sum().item()
                hop_total[t] += rel_path.size(0)

            if verbose:
                question_ids = questions['input_ids'].cpu().tolist()
                for i in range(rel_path.size(0)):
                    if not correct_per_hop[i].all():
                        tokens = data.tokenizer.convert_ids_to_tokens(question_ids[i])
                        print('================================================================')
                        print(' '.join(tokens))
                        print('> golden:     {}'.format([data.id2rel[r] for r in rel_path[i].tolist()]))
                        print('> prediction: {}'.format([data.id2rel[r] for r in pred_path[i].tolist()]))

    overall_acc = exact_match / count
    print(f"Exact path match: {overall_acc:.4f}")

    hop_accuracy = {
        "hop{}".format(t + 1): hop_correct[t] / hop_total[t]
        for t in sorted(hop_total)
    }
    hop_report = ["{}: {:.4f}".format(hop, acc) for hop, acc in hop_accuracy.items()]
    print("Per-hop relation accuracy: " + ", ".join(hop_report))

    return {'exact_match': overall_acc, 'hop_accuracy': hop_accuracy}
