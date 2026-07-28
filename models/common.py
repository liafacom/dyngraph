import json
import os
import random
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


DATASET_CLASS_SAMPLES = {"ohsumed": 9, "r8": 10, "agnews": 10, "snippets": 8, "dblp": 20}


def add_gaussian_noise(embeddings, noise_level=0.01):
    noise = torch.normal(0, noise_level, size=embeddings.size()).to(embeddings.device)
    return embeddings + noise


def _sample_ids(ids, labels, samples_per_class, seed):
    random_generator = random.Random(seed)
    grouped = {}
    for item_id, label in zip(ids, labels):
        grouped.setdefault(label, []).append(item_id)
    selected = []
    for item_ids in grouped.values():
        selected.extend(random_generator.sample(item_ids, min(samples_per_class, len(item_ids))))
    missing = len(grouped) * samples_per_class - len(selected)
    if missing:
        most_frequent = Counter(labels).most_common(1)[0][0]
        selected.extend(random_generator.choices(grouped[most_frequent], k=missing))
    selected_set = set(selected)
    remaining = [item_id for item_id in ids if item_id not in selected_set]
    return selected, remaining


def detailed_split(func, exp_number, train_size, val_size=1000, test_size=6960, **_):
    """Create the labeled, validation, unlabeled and test experiment partitions."""
    full, test, target_names, dataset_name = func()
    try:
        samples_per_class = DATASET_CLASS_SAMPLES[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from error
    train_ids, remaining_ids = _sample_ids(list(full.index), list(full["label"]), samples_per_class, exp_number)
    train = full.loc[train_ids].copy()
    remaining = full.loc[remaining_ids].copy()
    if train_size < len(train):
        moved = train.sample(len(train) - train_size, random_state=exp_number)
        remaining = pd.concat([remaining, moved])
        train = train.drop(moved.index)
    elif train_size > len(train):
        moved = remaining.sample(train_size - len(train), random_state=exp_number)
        train = pd.concat([train, moved])
        remaining = remaining.drop(moved.index)
    if val_size < len(remaining):
        validation = remaining.sample(val_size, random_state=exp_number)
        remaining = remaining.drop(validation.index)
        test = pd.concat([test, remaining])
    else:
        validation = remaining.copy()
    if test_size < len(test):
        removed = test.sample(len(test) - test_size, random_state=exp_number)
        test = test.drop(removed.index)
    complementary = pd.DataFrame()
    return full, train, complementary, validation, test, train_size, target_names, dataset_name


def generate_aug(frame, num_gen=2, repeats=1):
    """Generate keyboard and EDA variants for the labeled training examples."""
    import nltk
    import nlpaug.augmenter.char as char_augmenters
    import nlpaug.augmenter.word as word_augmenters
    import nlpaug.flow as flows

    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    augmenters = (
        char_augmenters.KeyboardAug(),
        flows.Sequential([
            flows.Sometimes([word_augmenters.RandomWordAug(action="swap")]),
            flows.Sometimes([word_augmenters.RandomWordAug(action="delete")]),
            flows.Sometimes([word_augmenters.SynonymAug(aug_src="wordnet")]),
        ]),
    )
    augmented = []
    for augmenter in augmenters:
        for _ in range(repeats):
            generated = frame.copy()
            generated["text_aug"] = generated["text"].apply(lambda text: augmenter.augment(text, num_gen))
            augmented.append(generated.explode("text_aug").reset_index(drop=True))
    return pd.concat(augmented).reset_index(drop=True)


def get_configs():
    path = "config.json"
    if not os.path.exists(path):
        return {"machine": "local", "url": None}
    with open(path, encoding="utf-8") as config_file:
        return json.load(config_file)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, "iloc") else self.texts[idx]
        if self.labels is None:
            return idx, text, None
        label = self.labels.iloc[idx] if hasattr(self.labels, "iloc") else self.labels[idx]
        return idx, text, label
