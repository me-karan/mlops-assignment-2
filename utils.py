import json
from typing import Any, Dict, Sequence

import torch
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import EvalPrediction


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: Sequence[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def get_label_maps(labels: Sequence[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def compute_metrics(prediction: EvalPrediction) -> Dict[str, float]:
    labels = prediction.label_ids
    predictions = prediction.predictions.argmax(-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': float(accuracy)}


def save_json(data: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)
