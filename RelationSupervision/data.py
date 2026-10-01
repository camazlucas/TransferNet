import torch
import csv
from transformers import AutoTokenizer
from utils.misc import invert_dict

STOP = "<stop>"


def build_rel2id(csv_files):
    # Vocabulario fechado de relacao: toda relacao que aparece na coluna 'paths'
    # dos arquivos passados (schema unificado do projeto: question,...,paths),
    # mais STOP para os hops que sobram. Nao amarrado a nenhum dataset especifico.
    relations = set()
    for fn in csv_files:
        with open(fn, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                relations.update(row['paths'].split('|'))

    relations = sorted(relations)
    return {r: i for i, r in enumerate(relations + [STOP])}


def collate(batch):
    batch = list(zip(*batch))
    question, rel_path = batch
    question = {k: torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    rel_path = torch.stack(rel_path)
    return question, rel_path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __getitem__(self, index):
        return self.questions[index]

    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, fn, tokenizer_name, rel2id, max_hops, batch_size, training=False):
        print('Reading questions from {}'.format(fn))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=True
        )
        self.rel2id = rel2id
        self.id2rel = invert_dict(rel2id)
        stop_id = rel2id[STOP]

        data = []
        with open(fn, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = self.tokenizer(
                    row['question'].strip(),
                    max_length=64,
                    padding='max_length',
                    return_tensors="pt"
                )

                rel_names = row['paths'].split('|')
                rel_ids = [rel2id[r] for r in rel_names][:max_hops]
                rel_ids += [stop_id] * (max_hops - len(rel_ids))

                data.append((question, torch.LongTensor(rel_ids)))

        print('data number: {}'.format(len(data)))

        dataset = Dataset(data)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )


def load_data(
    input_train,
    input_valid,
    tokenizer_name,
    batch_size,
    max_hops,
    rel2id
):
    train_data = DataLoader(
        input_train,
        tokenizer_name,
        rel2id,
        max_hops,
        batch_size,
        training=True
    )

    valid_data = DataLoader(
        input_valid,
        tokenizer_name,
        rel2id,
        max_hops,
        batch_size
    )

    return train_data, valid_data
