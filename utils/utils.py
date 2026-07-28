import json
from pathlib import Path

import pandas as pd
import requests
from sklearn import preprocessing


DATA_COLUMNS = ["text", "label", "label_names", "subset"]
MAX_LENGTHS = {
    "ohsumed": 288,
    "r8": 256,
    "agnews": 64,
    "snippets": 64,
    "dblp": 32,
}

SETUP_C_SIZES = {
    0.01: {"ohsumed": (207, 6193), "r8": (80, 6594), "agnews": (40, 6960), "snippets": (64, 11276), "dblp": (120, 22880)},
    0.05: {"ohsumed": (370, 6030), "r8": (383, 6291), "agnews": (400, 6600), "snippets": (617, 10723), "dblp": (1200, 21800)},
    0.1: {"ohsumed": (740, 5660), "r8": (767, 5907), "agnews": (800, 6200), "snippets": (1234, 10106), "dblp": (2400, 20600)},
    0.2: {"ohsumed": (1480, 4920), "r8": (1534, 5139), "agnews": (1600, 5400), "snippets": (2468, 8872), "dblp": (4800, 18200)},
}


def send_msg(content, user="User", avatar_url="", thread_name="", url=None):
    """Send an optional experiment notification to a Discord-compatible webhook."""
    if not url:
        return
    message = {"content": content, "username": user, "avatar_url": avatar_url, "thread_name": thread_name}
    try:
        response = requests.post(url, data=json.dumps(message), headers={"Content-Type": "application/json"}, timeout=10)
        if response.status_code != 204:
            print(f"Failed to send notification: HTTP {response.status_code}")
    except requests.RequestException as error:
        print(f"Failed to send notification: {error}")


def check_log_path(dataset_name):
    Path("artifacts/logs", dataset_name).mkdir(parents=True, exist_ok=True)


def _load_text_meta_dataset(folder, filename, dataset_name, train_subset):
    data = pd.read_csv(Path(folder) / f"{filename}.txt", encoding="latin-1", header=None, delimiter="\t", names=["text"])
    meta = pd.read_csv(Path(folder) / f"{filename}.meta", header=None, delimiter="\t")
    data["class"] = meta[2]
    data["subset"] = meta[1]
    encoder = preprocessing.LabelEncoder()
    data["label"] = encoder.fit_transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].astype(str)
    train = data[data["subset"] == train_subset].copy()
    test = data[data["subset"] == "test"].copy()
    return train, test, encoder.classes_, dataset_name


def get_ohsumed(folder="datasets/"):
    """Load the Ohsumed corpus distributed with the repository."""
    return _load_text_meta_dataset(folder, "ohsumed", "ohsumed", "training")


def get_r8(folder="datasets/"):
    """Load the R8 corpus distributed with the repository."""
    return _load_text_meta_dataset(folder, "R8", "r8", "train")


def get_snippets(folder="datasets/"):
    """Load the Web Snippets train and test files."""
    def load_subset(name):
        frame = pd.read_csv(Path(folder) / "data-web-snippets" / f"{name}.txt", header=None)
        values = frame[0].astype(str)
        frame["text"] = values.map(lambda value: " ".join(value.split()[:-1]))
        frame["label_names"] = values.map(lambda value: value.split()[-1])
        frame["subset"] = name
        return frame

    train = load_subset("train")
    test = load_subset("test")
    encoder = preprocessing.LabelEncoder()
    train["label"] = encoder.fit_transform(train["label_names"])
    test["label"] = encoder.transform(test["label_names"])
    return train, test, list(encoder.classes_), "snippets"


def get_dblp(folder="datasets/"):
    """Load the DBLP corpus distributed with the repository."""
    text = pd.read_csv(Path(folder) / "dblp.txt", header=None, names=["text"])
    labels = pd.read_csv(Path(folder) / "dblp_labels.txt", header=None, delimiter="\t", names=["idx", "subset", "label_names"])
    data = pd.concat([text, labels], axis=1)
    encoder = preprocessing.LabelEncoder()
    data["label"] = encoder.fit_transform(data["label_names"])
    target_names = [str(label) for label in encoder.classes_]
    train = data[data["subset"] == "train"].reset_index(drop=True)
    test = data[data["subset"] == "test"].reset_index(drop=True)
    return train[DATA_COLUMNS], test[DATA_COLUMNS], target_names, "dblp"


def get_agnew():
    """Download and load AG News through torchtext."""
    import torchtext

    rows = [(text, label, subset) for subset in ("train", "test") for label, text in torchtext.datasets.AG_NEWS(split=subset)]
    data = pd.DataFrame(rows, columns=["text", "label", "subset"])
    data["label_names"] = data["label"]
    data["label"] -= 1
    train = data[data["subset"] == "train"]
    test = data[data["subset"] == "test"]
    return train[DATA_COLUMNS], test[DATA_COLUMNS], ["2", "3", "1", "0"], "agnews"


def best_max_length():
    return MAX_LENGTHS.copy()


def choose_setup(name, dataset, perc=0.01):
    """Return sampling settings for the supported experiment setups."""
    if dataset not in MAX_LENGTHS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if name == "C":
        try:
            train_size, test_size = SETUP_C_SIZES[perc][dataset]
        except KeyError as error:
            raise ValueError(f"Unsupported training percentage: {perc}") from error
        return {"proj": "TC-Semisup-Setup-C", "join_train_test": True, "train_size_per_class": False, "sampling_train": train_size, "compl_size": 0, "val_size": 1000, "test_size": test_size}
    if name == "A":
        if dataset != "snippets":
            raise ValueError("Setup A is only configured for snippets")
        return {"proj": "TC-Semisup-Setup-A", "join_train_test": True, "train_size_per_class": True, "sampling_train": 20, "compl_size": 1000, "val_size": 160, "test_size": 11020}
    if name == "B":
        if dataset != "agnews":
            raise ValueError("Setup B is only configured for agnews")
        return {"proj": "TC-Semisup-Setup-B", "join_train_test": True, "train_size_per_class": True, "sampling_train": 10, "compl_size": 0, "val_size": 1000, "test_size": 6960}
    raise ValueError(f"Unsupported setup: {name}")
