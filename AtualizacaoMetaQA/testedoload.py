from .data import load_data
from utils.misc import invert_dict

ent2id, rel2id, triples, train_data, test_data = load_data(
    "Dataset\\MetaQA",
    "bert-base-uncased",
    16
)

print(len(ent2id))
print(len(rel2id))
print(triples.shape)

batch = next(iter(train_data))

print(batch[0].shape)
print(batch[2].shape)
print(batch[3].shape)