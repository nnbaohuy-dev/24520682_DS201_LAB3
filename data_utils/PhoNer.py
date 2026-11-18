import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

import json
import string


def collate_fn(items: list[dict]) -> dict:
    input_ids =  [item["input_ids"] for item in items]
    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    labels = [item["labels"] for item in items]
    label_lengths  = torch.tensor([label.shape[0] for label in labels], dtype=torch.long)
    max_label_len = int(label_lengths.max())

    input_ids = torch.stack(
        [
            F.pad(
                input,
                pad=(0, max_len - input.shape[0]),
                mode="constant",
                value=0
            ) for input in input_ids
        ]
    )

    labels = torch.stack(
        [
            F.pad(
                label,
                pad=(0, max_label_len - label.shape[0]),
                mode="constant",
                value=-1
            ) for label in labels
        ]
    )

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": labels,
    }


class Vocab:
    def __init__(self, train_path: str, word_key: str='words', label_key: str='tags'):
        self.pad = "<pad>"
        self.unk = "<unknow>"
        self.word_key = word_key
        self.label_key  = label_key

        with open(train_path, encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        all_words = set()
        labels = set()

        for item in data:
            words = self.preprocess_words(item[self.word_key])
            all_words.update(words)
            labels.update(item[self.label_key])
        
        self.w2i = {
            self.pad : 0,
            self.unk : 1,
        }

        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx

        self.l2i = {}
        for idx, label in enumerate(sorted(labels)):
            self.l2i[label] = idx

        self.i2w = {idx: word for idx, word in self.w2i.items()}

        self.i2l = {idx: label for idx, label in self.l2i.items()}

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    @property
    def len(self) -> int:
        return len(self.w2i)

    def preprocess_words(self, words: list[str]) -> list[str]:
        translator = str.maketrans("", "", string.punctuation)
        processed_words = []
        for word in words:
            word = word.lower()
            word = word.translate(translator)
            processed_words.append(word)

        return processed_words

    def encode_sentence(self, words: list[str]) -> torch.Tensor:
        words = self.preprocess_words(words)

        if len(words) == 0:
            words = self.unk

        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_id) for word in words]

        return torch.tensor(word_ids, dtype=torch.long)

    def encode_labels(self, labels: list[str]) -> torch.Tensor:
        label_ids =  [self.l2i[label] for label in labels]
        return torch.tensor(label_ids, dtype=torch.long)

    def decode_labels(self, label_ids: torch.Tensor) -> list[str]:
        label_ids = label_ids.tolist()
        labels = [self.i2l[idx] for idx in label_ids]

        return labels

class PhoNer(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        with open(path, encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        words = item[self.vocab.word_key]
        labels = item[self.vocab.label_key]

        input_ids = self.vocab.encode_sentence(words)
        labels_idx = self.vocab.encode_labels(labels)

        return {
            "input_ids": input_ids,
            "labels": labels_idx
        }