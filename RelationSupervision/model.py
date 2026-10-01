import torch
import torch.nn as nn
import math
from transformers import AutoModel
from utils.BiGRU import GRU, BiGRU

class TransferNet(nn.Module):
    def __init__(self, args, rel2id):
        super().__init__()
        self.args = args
        self.max_hops = getattr(args, "max_hops", 4)
        num_relations = len(rel2id)

        self.bert_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = []
        for i in range(self.max_hops):
            m = nn.Sequential(
                nn.Linear(dim_hidden * 2, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

        self.rel_classifier = nn.Linear(dim_hidden, num_relations)


    def forward(self, questions, rel_path=None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        word_attns = []
        rel_probs = []
        rel_logits = []
        last_h = torch.zeros_like(q_embeddings)
        for t in range(self.max_hops):
            cq_t = self.step_encoders[t](torch.cat((q_embeddings, last_h), dim=1)) # [bsz, dim_h], considera o hop anterior
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
            q_dist = q_dist * questions['attention_mask'].float()
            q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
            word_attns.append(q_dist)
            ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]
            ctx_h = ctx_h + cq_t
            last_h = ctx_h

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            rel_logits.append(rel_logit)
            rel_probs.append(torch.softmax(rel_logit, dim=1))

        if self.training:
            # rel_path: [bsz, max_hops], gold relation id per hop (STOP fills hops past the path's end)
            loss = sum(
                nn.functional.cross_entropy(rel_logits[t], rel_path[:, t])
                for t in range(self.max_hops)
            ) / self.max_hops
            return {'loss': loss}

        return {
            'word_attns': word_attns,
            'rel_probs': rel_probs,
        }
