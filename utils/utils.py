import glob
import json
import os
from datetime import datetime
import pytz
from typing import Counter
from datasets import load_dataset

import nltk
from nltk.corpus import reuters
import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Definir as stopwords em português (ou altere para o idioma do seu dataset)
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Exemplo de DataFrame com dados de textos e rótulos
# Suponha que o DataFrame esteja na variável 'df' e as features de texto estejam na coluna 'text' e os rótulos na coluna 'label'.
# Vamos considerar que você já pré-processou os dados de texto anteriormente (por exemplo, tokenização, remoção de stop words, etc.).
# E também vamos considerar que os rótulos são numéricos (0, 1, 2, etc.).
# Se necessário, adapte essa parte de acordo com o seu DataFrame real.

TEST_SIZE = 0.3
RANDOM_STATE = 0

def send_msg(content, user="User", avatar_url="", thread_name="", url=None, testing=False):
    if not url:
        return
    webhook_url = url

    mensagem = {
        "content": content,
        "username": user,  # Opcional: personaliza o nome do usuário do webhook
        "avatar_url": avatar_url,  # Opcional: URL de uma imagem para o avatar do webhook
        "thread_name": thread_name,
    }
    try:
        # Envia a mensagem
        response = requests.post(
            webhook_url,
            data=json.dumps(mensagem),
            headers={"Content-Type": "application/json"},
        )

        # Verifica se a requisição foi bem-sucedida
        if response.status_code == 204:
            print("Mensagem enviada com sucesso!")
        else:
            print("Falha ao enviar mensagem. Msg:")
            print(str(mensagem))
    except Exception as e:
        print(e)

def writing_file_sync(run, filename="wandb_sync_file_imodel.txt"):
    dir = run.dir
    parent_dir = os.path.dirname(dir)
    # Define o fuso horário de Campo Grande, MS
    campo_grande_tz = pytz.timezone('America/Campo_Grande')
    # Obtém a data e hora atual no fuso horário UTC
    now_utc = datetime.now(pytz.utc)
    # Converte para o fuso horário de Campo Grande, MS
    now_campo_grande = now_utc.astimezone(campo_grande_tz)
    # Formata a data e hora com o dia da semana
    formatted_date_time = now_campo_grande.strftime('%A, %d/%m/%Y %H:%M:%S')

    # print(formatted_date_time)
    with open(filename, "a") as arquivo:
        arquivo.write(f"## {formatted_date_time} - {run.config.dataset_name}\n")
        arquivo.write(f"wandb sync {parent_dir}\n")

def get_classic4():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/classic4.csv")
    # test_classic4_size = 1419
    test_classic4_size = TEST_SIZE
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = le.transform(data["class"])
    df_train, df_test = train_test_split(
        data, test_size=test_classic4_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "classic4"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name


def get_20newsgroups():
    data_train = fetch_20newsgroups(subset="train")
    data_test = fetch_20newsgroups(subset="test")
    target_names = data_test.target_names  # type: ignore

    def get_df(data, subset):
        df = pd.DataFrame(data.data, columns=["text"])
        df["label"] = data.target
        df["subset"] = subset
        df["label_names"] = [target_names[i] for i in df.label]
        return df

    df = pd.concat(
        [get_df(data_train, "train"), get_df(data_test, "test")], ignore_index=True
    )
    df_train = df[df.subset == "train"].copy() #get_df(data_train, "train")
    df_test = df[df.subset == "test"].copy() #get_df(data_test, "test")
    dataset_name = "20newsgroups"
    return df_train, df_test, target_names, dataset_name

def check_log_path(dataset_name):
    caminho = f"artifacts/logs/{dataset_name}"
    if not os.path.exists(caminho):
        os.makedirs(caminho)
    
def get_bbc(pathto="datasets/"):
    data = pd.read_csv(pathto + "bbc.csv")
    data["class"] = data.label
    # total 300
    test_cstr_size = TEST_SIZE
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data = data.reset_index()
    df_train, df_test = train_test_split(
        data, test_size=test_cstr_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "bbc"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_bbcsport(pathto="datasets/"):
    data = pd.read_csv(pathto + "bbcsport.csv")
    # total 300
    test_cstr_size = TEST_SIZE
    le = preprocessing.LabelEncoder()
    data["class"] = data.label
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data = data.reset_index()
    df_train, df_test = train_test_split(
        data, test_size=test_cstr_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "bbcsport"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name


def get_pge(pathto="datasets/"):
    # d = pd.read_csv("datasets/pge.csv")
    # d.loc[d.label_names.isin(["penhora", "arresto"]),"label_names"] = "penhora_arresto"
    # d.label_names.value_counts()
    # d.to_csv("datasets/pge.csv", index=False)
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv(f"{pathto}/pge.csv")
    # test_classic4_size = 1419
    test_size = 0.2
    le = preprocessing.LabelEncoder()
    le.fit(data["label_names"])
    data["label"] = le.transform(data["label_names"])
    df_train, df_test = train_test_split(
        data, test_size=test_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "pge"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_cstr(pathto="datasets/"):
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv(pathto + "CSTR.csv")
    # total 300
    test_cstr_size = TEST_SIZE
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data = data.reset_index()
    df_train, df_test = train_test_split(
        data, test_size=test_cstr_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "cstr"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_trec(folder="datasets/", fine=False):
    dataset_name = "trec"
    
    df_train = pd.read_csv(f"{folder}trec_train.csv")
    df_test = pd.read_csv(f"{folder}trec_test.csv")
    df_train["text"] = [str(t) for t in df_train.text.values]
    df_train["subset"] = "train"
    df_test["text"] = [str(t) for t in df_test.text.values]
    df_test["subset"] = "test"
    
    if fine:
        df_train["label"] = df_train["label-fine"]
        df_test["label"] = df_test["label-fine"]
    else:
        df_train["label"] = df_train["label-coarse"]
        df_test["label"] = df_test["label-coarse"]
    df_train["label_names"] = df_train["label"]
    df_test["label_names"] = df_test["label"]
    
    target_names = list(df_train.label_names.unique())
    
    return df_train, df_test, target_names, dataset_name

def get_trec_fine(folder="datasets/"):
    df_train, df_test, target_names, _ = get_trec(folder, fine=True)
    dataset_name = "trec_fine"
    return df_train, df_test, target_names, dataset_name
    

def get_ohsumed(folder="datasets/"):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    data = pd.read_csv(
        f"{folder}ohsumed.txt", encoding="latin-1", header=None, delimiter="\t"
    )
    data.columns = ["text"]
    meta = pd.read_csv(f"{folder}ohsumed.meta", header=None, delimiter="\t")
    data["class"] = meta[2]  # labels
    data["subset"] = meta[1]  # subset
    # total 7400
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    # print(data.groupby(['subset', 'label_names']).count())
    df_train = data[data.subset == "training"].copy()
    df_test = data[data.subset == "test"].copy()
    dataset_name = "ohsumed"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

 
def get_snippets(pathto="datasets/"):
    dataset_name = "snippets"
    
    df_train = pd.read_csv(f"{pathto}data-web-snippets/train.txt", header=None)
    df_train["text"] = [" ".join(t.split()[:-1])  for t in df_train[0]]
    df_train["label_names"] = [t.split()[-1]  for t in df_train[0]]
    df_train["subset"] = "train"
    
    df_test = pd.read_csv(f"{pathto}data-web-snippets/test.txt", header=None)
    df_test["text"] = [" ".join(t.split()[:-1])  for t in df_test[0]]
    df_test["label_names"] = [t.split()[-1]  for t in df_test[0]]
    df_test["subset"] = "test"    
    
    le = preprocessing.LabelEncoder()
    df_train["label"] = le.fit_transform(df_train["label_names"])
    df_test["label"] = le.transform(df_test["label_names"])
    
    target_names = le.classes_
    return df_train, df_test, list(target_names), dataset_name

def get_r8_double(folder="datasets/"):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    # data = pd.read_csv(
    #     f"{folder}R8.txt", encoding="latin-1", header=None, delimiter="\t"
    # )
    # data.columns = ["text"]
    # meta = pd.read_csv(f"{folder}R8.meta", header=None, delimiter="\t")
    # data["class"] = meta[2]  # labels
    # data["subset"] = meta[1]  # subset
    data = pd.read_csv(f"{folder}/R8_double.csv")
    # total 7400
    le = preprocessing.LabelEncoder()
    le.fit(data["tlabel"])
    data["label"] = le.transform(data["tlabel"])
    data["label_names"] = data["tlabel"]
    data["text"] = data.text.apply(str)
    # print(data.groupby(['subset', 'label_names']).count())
    df_train = data[data.subset == "train"].copy()
    df_test = data[data.subset == "test"].copy()
    dataset_name = "r8_double"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_reuters():
    nltk.download('reuters')
    documents = reuters.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    
    # Transform multilabel labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id]) 
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])
    df_train = pd.DataFrame({"text": train_docs})
    df_train["label"] =  [l for l in train_labels]
    df_test = pd.DataFrame({"text": test_docs})
    df_test["label"] = [l for l in test_labels]
    target_names = mlb.classes_
    dataset_name = "reuters"
    return df_train, df_test, target_names, dataset_name

def get_r8(folder="datasets/"):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    data = pd.read_csv(
        f"{folder}R8.txt", encoding="latin-1", header=None, delimiter="\t"
    )
    data.columns = ["text"]
    meta = pd.read_csv(f"{folder}R8.meta", header=None, delimiter="\t")
    data["class"] = meta[2]  # labels
    data["subset"] = meta[1]  # subset
    # total 7400
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data.text.apply(str)
    # print(data.groupby(['subset', 'label_names']).count())
    df_train = data[data.subset == "train"].copy()
    df_test = data[data.subset == "test"].copy()
    dataset_name = "r8"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_r8_tiny(folder="datasets/", max_sample_class=10, random_state=42):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    data = pd.read_csv(
        f"{folder}R8.txt", encoding="latin-1", header=None, delimiter="\t"
    )
    data.columns = ["text"]
    meta = pd.read_csv(f"{folder}R8.meta", header=None, delimiter="\t")
    data["class"] = meta[2]  # labels
    data["subset"] = meta[1]  # subset
    # total 7400
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data.text.apply(str)
    print(data.groupby(['subset', 'label_names']).count())
    df_train = data[data.subset == "train"].copy()
    
    # Balanceado (50 exemplos por classe)
    df_train = df_train.groupby("label_names", group_keys=False).apply(
        lambda x: x.sample(n=min(max_sample_class, len(x)), random_state=random_state)
    ).reset_index(drop=True)
    df_test = data[data.subset == "test"].copy()
    
    # Balanceado (50 exemplos por classe)
    df_test = df_test.groupby("label_names", group_keys=False).apply(
        lambda x: x.sample(n=min(max_sample_class, len(x)), random_state=random_state)
    ).reset_index(drop=True)
    dataset_name = "r8_tiny"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name

def get_r52(folder="datasets/"):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    data = pd.read_csv(
        f"{folder}R52.txt", encoding="latin-1", header=None, delimiter="\t"
    )
    data.columns = ["text"]
    meta = pd.read_csv(f"{folder}R52.meta", header=None, delimiter="\t")
    data["class"] = meta[2]  # labels
    data["subset"] = meta[1]  # subset
    # total 7400
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    # print(data.groupby(['subset', 'label_names']).count())
    df_train = data[data.subset == "train"].copy()
    df_test = data[data.subset == "test"].copy()
    dataset_name = "r52"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name


def get_mr(folder="datasets/"):
    # Dataset source: https://github.com/vitormeriat/nlp-based-text-gcn/tree/main/data/corpus
    data = pd.read_csv(
        f"{folder}mr.txt", encoding="latin-1", header=None, delimiter="\t"
    )
    target_names = ["pos", "neg"]
    data.columns = ["text"]
    meta = pd.read_csv(f"{folder}mr.meta", header=None, delimiter="\t")
    data["label"] = meta[2]  # labels
    data["subset"] = meta[1]  # subset
    data["label_names"] = [target_names[i] for i in data.label]
    # total 10662
    df_train = data[data.subset == "train"].copy()
    df_test = data[data.subset == "test"].copy()
    dataset_name = "movie_review"
    return df_train, df_test, target_names, dataset_name

def make_scores_json(y_true, y_pred, subset, target_names=None):
    json_metrics = {}
    # acc, f1-micro, macro, prec, recal
    # json_metrics[f"acc_{subset}"] = 
    return json_metrics

def make_scores(y_true, y_pred, pred_proba, target_names, subset):
    y_true = np.array(y_true)
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    print(len(y_true), pred_proba.shape)
    multi_class = "ovr"
    if pred_proba.shape[1] == 2:
        multi_class = "raise"
        pred_proba = pred_proba[:, 0]
        y_true_roc = y_true
    else:
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(np.concatenate([y_pred, y_true]).reshape(-1, 1))  # type: ignore
        y_true_roc = encoder.transform(y_true.reshape(-1, 1)).todense()
        y_true_roc = np.asarray(y_true_roc)  # type: ignore
    auc_macro = roc_auc_score(
        y_true_roc,
        pred_proba,
        multi_class=multi_class,
        average="macro",
    )
    # auc_micro = roc_auc_score(
    #    y_true_roc, pred_proba,
    #    multi_class=multi_class,
    #    average="micro",
    # )
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    return {
        f"size_{subset}": len(y_pred),
        f"metric_accuracy_{subset}": report["accuracy"],  # type: ignore
        f"metric_precision_macro_{subset}": report["macro avg"]["precision"],  # type: ignore
        f"metric_recall_macro_{subset}": report["macro avg"]["recall"],  # type: ignore
        f"metric_f1score_macro_{subset}": report["macro avg"]["f1-score"],  # type: ignore
        # f'metric_auc_ovr_micro_{subset}': auc_micro,
        f"metric_auc_ovr_macro_{subset}": auc_macro,
        f"report_{subset}": report,
        # 'metric_weighted_precision': report['weighted avg']['precision'],
        # 'metric_weighted_recall': report['weighted avg']['recall'],
        # 'metric_weighted_f1-score': report['weighted avg']['f1-score'],
        # 'avg_expected_loss': avg_expected_loss,
        # 'avg_bias': avg_bias,
        # 'avg_var': avg_var
    }


columns_data = ["text", "label", "label_names", "subset"]


def get_nsf():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/NSF.csv")
    data["text"] = data.x
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["subset"] = data.fold
    df_train = data[columns_data][(data.fold == "train") | (data.fold == "val")]
    df_test = data[columns_data][(data.fold == "test")]
    dataset_name = "nsf"
    target_names = le.classes_
    return df_train, df_test, target_names, dataset_name


def get_syskillwebert():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/SyskillWebert.csv")
    # print(data)
    test_syskillwebert_size = TEST_SIZE
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=test_syskillwebert_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "syskillwebert"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_twitter():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    dn = [
        "mapping.txt",
        "test_labels.txt",
        "test_text.txt",
        "train_labels.txt",
        "train_text.txt",
        "val_labels.txt",
        "val_text.txt",
    ]

    path_data = "datasets/twitter"

    if not os.path.exists(path_data):
        os.makedirs(path_data)
    files = {}

    for d in dn:
        with open(path_data + "/" + d, "r") as f:
            files[d] = f.readlines()
    map_lab = {0: "negative", 1: "neutral", 2: "positive"}
    labels = []
    sizes = {}
    for f in ["test_labels.txt", "train_labels.txt", "val_labels.txt"]:
        sizes[f] = len(files[f])
        labels.extend(files[f])
    labels = [map_lab[int(l)] for l in labels]
    texts = []
    for f in ["test_text.txt", "train_text.txt", "val_text.txt"]:
        texts.extend(files[f])
    texts = [t.replace("\n", " ") for t in texts]
    subsets = (
        ["test"] * sizes["test_labels.txt"]
        + ["train"] * sizes["train_labels.txt"]
        + ["train"] * sizes["val_labels.txt"]
    )
    data = {"text": texts, "class": labels, "subset": subsets}
    data = pd.DataFrame(data)
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    df_train = data[data.subset == "train"]
    df_test = data[data.subset == "test"]
    dataset_name = "twitter"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_webkb():
    # 70% train
    # 30% test
    text = pd.read_csv("datasets/WebKB.txt", header=None)
    text.columns = ["text"]
    labels = pd.read_csv("datasets/WebKB_label.txt", header=None, delimiter="\t")
    labels.columns = ["idx", "subset", "label_names"]
    df = pd.concat([text, labels], axis=1)
    le = preprocessing.LabelEncoder()
    le.fit(df["label_names"])
    target_names = le.classes_
    df["label"] = le.transform(df["label_names"])
    
    df_train = df[df.subset=="train"].copy()
    df_train.reset_index(drop=True, inplace=True)
    df_test = df[df.subset=="test"].copy()
    df_test.reset_index(drop=True, inplace=True)
    dataset_name = "webkb"
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name  # type: ignore



def get_dblp():
    # 70% train
    # 30% test
    text = pd.read_csv("datasets/dblp.txt", header=None)
    text.columns = ["text"]
    labels = pd.read_csv("datasets/dblp_labels.txt", header=None, delimiter="\t")
    labels.columns = ["idx", "subset", "label_names"]
    df = pd.concat([text, labels], axis=1)
    le = preprocessing.LabelEncoder()
    le.fit(df["label_names"])
    target_names = [str(c) for c in le.classes_]
    df["label"] = le.transform(df["label_names"])
    
    df_train = df[df.subset=="train"].copy()
    df_train.reset_index(drop=True, inplace=True)
    df_test = df[df.subset=="test"].copy()
    df_test.reset_index(drop=True, inplace=True)
    dataset_name = "dblp"
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name  # type: ignore

def get_ag_news():
    path_data = "datasets/ag_news"
    df_train = pd.read_csv(f"{path_data}/train.csv", header=None)
    df_train.columns = ["label", "title", "content"]
    df_test = pd.read_csv(f"{path_data}/test.csv", header=None)
    df_test.columns = ["label", "title", "content"]
    df_train["text"] = df_train.title + " " + df_train.content
    df_train["subset"] = "train"
    df_test["text"] = df_test.title + " " + df_test.content
    df_test["subset"] = "test"
    labels = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    df_train["label_names"] = [labels[l] for l in df_train.label]
    df_test["label_names"] = [labels[l] for l in df_test.label]
    target_names = list(labels.values())
    dataset_name = "ag_news"
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_imdb():
    # Dataset source:
    # !wget -O $path_data/aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    # path_data = "datasets/imdb"
    # !tar -xvf $path_data/aclImdb_v1.tar.gz -C $path_data
    neg_files_train = []
    neg_files_test = []

    neg_files_test.extend(glob.glob("datasets/imdb/aclImdb/test/neg/*.txt"))
    neg_files_train.extend(glob.glob("datasets/imdb/aclImdb/train/neg/*.txt"))

    pos_files_train = []
    pos_files_test = []
    pos_files_test.extend(glob.glob("datasets/imdb/aclImdb/test/pos/*.txt"))
    pos_files_train.extend(glob.glob("datasets/imdb/aclImdb/train/pos/*.txt"))

    texts = []
    for path in neg_files_test:
        with open(path, "r") as f:
            texts.append([f.read(), "negative", "test"])
    for path in neg_files_train:
        with open(path, "r") as f:
            texts.append([f.read(), "negative", "train"])
    for path in pos_files_test:
        with open(path, "r") as f:
            texts.append([f.read(), "positive", "test"])
    for path in pos_files_train:
        with open(path, "r") as f:
            texts.append([f.read(), "positive", "train"])
    data = pd.DataFrame(texts, columns=["text", "class", "subset"])
    # total 50000
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    df_train = data[data.subset == "train"]
    df_test = data[data.subset == "test"]
    dataset_name = "imdb"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_stats(df_train, df_test, target_names, dataset_name):
    print(f"Analisando dataset: {dataset_name}")
    dataset = pd.concat([df_train, df_test])
    dic = {}
    dic["dataset"] = dataset_name
    dic["#train"] = len(df_train)
    dic["#test"] = len(df_test)
    dic["#total"] = len(dataset)
    dic["#classes"] = dataset.label.nunique()
    
    def get_class_coef(y, tag):
                # Contagem das classes
        class_counts = Counter(y)
        total_samples = len(y)

        # Coeficiente de Gini
        gini = 1 - sum((count / total_samples) ** 2 for count in class_counts.values())
        # print("Coeficiente de Gini:", gini)
        dic = {f"coef gini({tag})": gini}

        # Índice de Shannon
        shannon_index = -sum((count / total_samples) * np.log(count / total_samples) for count in class_counts.values())
        # print("Índice de Shannon:", shannon_index)
        dic[f"shannon index({tag})"] = shannon_index

        # Coeficiente de Variância (CV)
        mean_class_size = np.mean(list(class_counts.values()))
        std_class_size = np.std(list(class_counts.values()))
        cv = std_class_size / mean_class_size
        # print("Coeficiente de Variância:", cv)
        dic[f"coef var({tag})"] = cv
        return dic
            
    # Função para calcular o tamanho médio do texto
    def avg_text_length(text):
        return len(text.split())

    # Função para calcular a quantidade média de textos por classe
    def avg_texts_per_class(data, column_name):
        return data.groupby(column_name)["text"].count().mean()

    # Função para calcular a quantidade de palavras distintas
    def distinct_words(text):
        words = word_tokenize(text)
        return len(set(words))

    # Adicionar coluna com tamanho do texto
    dataset["text_length"] = dataset["text"].apply(avg_text_length)
    
    dic["#tokens max"] = dataset.text_length.max()
    dic["#tokens min"] = dataset.text_length.min()

    # Adicionar coluna com a quantidade de palavras distintas
    dataset["distinct_words"] = dataset["text"].apply(distinct_words)

    dataset["words"] = dataset.text.str.split().map(lambda x: len(x))

    # Calcular o tamanho médio do texto
    avg_text_length = dataset["text_length"].mean()  # type: ignore

    # Calcular percentis
    percentil_90 = np.percentile(dataset["text_length"], 90)
    percentil_95 = np.percentile(dataset["text_length"], 95)
    dic["#length 90p"] = percentil_90
    dic["#length 95p"] = percentil_95
    
    # Calcular a quantidade média de textos por classe
    avg_texts_per_train = avg_texts_per_class(dataset, "label")

    # Exibir as estatísticas
    dic["#token mean"] = avg_text_length
    # Calcular a quantidade média de palavras distintas
    avg_distinct_words = dataset["distinct_words"].mean()
    dic["#mean distinct words"] = avg_distinct_words
    dic["#mean distinct words per doc"] = dataset.words.mean()
    
    dataset["char_len"] = dataset["text"].str.len()
    dic["#mean char"] = dataset["char_len"].mean() 
    dic["#std char"] = dataset["char_len"].std() 
    
    dic["#mean docs per class"] = avg_texts_per_train
    dic["#std docs per class"] = (
        dataset.groupby("label_names")["text"].count().std()
    )

    # Se quiser, você também pode calcular a média por conjunto (treino/teste)
    avg_texts_per_subset = avg_texts_per_class(dataset, "subset")
    dic["#mean docs per subset"] = avg_texts_per_subset
    dic["#std docs per subset"] = (
        dataset.groupby("subset")["text"].count().std()
    )
    
    dic.update(get_class_coef(df_train.label.tolist(), "train"))
    dic.update(get_class_coef(dataset.label.tolist(), "all"))

    dic["classes"] = ", ".join([str(t) for t in target_names])
    dic["#samples per class"] = (
        dataset.groupby("label_names")["text"].count().to_dict()
    )

    return dic

def get_trec_6():

    # Carregar o dataset
    dataset = load_dataset("CogComp/trec")
    
    # Função para transformar o dataset em um DataFrame
    def convert_to_dataframe(hf_dataset, subset_name):
        df = pd.DataFrame({
            'text': hf_dataset['text'],
            'label': hf_dataset['coarse_label'],
            'subset': subset_name
        })
        return df

    target_names = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
    # Converter os subsets de treino e teste para DataFrames
    train_df = convert_to_dataframe(dataset['train'], 'train')
    train_df["label_names"] = [target_names[l] for l in train_df["label"]]
    test_df = convert_to_dataframe(dataset['test'], 'test')
    test_df["label_names"] = [target_names[l] for l in test_df["label"]]
    # test_df["label"] = test_df["label"]-1
    
    dataset_name = "TREC6"
    # Concatenar os DataFrames
    

    return train_df, test_df, target_names, dataset_name


def get_agnew():
    import torchtext
    data_train = torchtext.datasets.AG_NEWS(split='train')
    data_test = torchtext.datasets.AG_NEWS(split='test')
    data = []
    for label, text in data_train:
        data.append([text, label, "train"])
    for label, text in data_test:
        data.append([text, label, "test"])
    
    data = pd.DataFrame(data, columns=["text", "label", "subset"])
    data['label_names'] = data['label']
    data['label'] = data['label']-1
    df_train = data[data.subset == "train"]
    df_test = data[data.subset == "test"]
    dataset_name = "agnews"
    target_names = ["2", "3", "1", "0"]
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name

def get_dmozcomputers():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/dmoz_computers.csv")
    # print(data)
    test_dmoz_computers_size = TEST_SIZE
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=test_dmoz_computers_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "dmozcomputers"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_dmozscience():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/dmoz_science.csv")
    # print(data)
    test_dmoz_computers_size = TEST_SIZE
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=test_dmoz_computers_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "dmozscience"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_dmozhealth():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/dmoz_health.csv")
    # print(data)
    test_dmoz_computers_size = TEST_SIZE
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=test_dmoz_computers_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "dmozhealth"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_dmozsports():
    # Dataset source: https://github.com/ragero/text-collections/tree/master/complete_texts_csvs
    data = pd.read_csv("datasets/dmoz_sports.csv")
    # print(data)
    test_dmoz_computers_size = TEST_SIZE
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["class"])
    data["label"] = le.transform(data["class"])
    data["label_names"] = data["class"]
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=test_dmoz_computers_size, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "dmozsports"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name


def get_toy_data():

    # Gerando um dataset textual com 3 classes: Economia, Esporte e Ciências.
    # Cada classe terá de 2 a 4 exemplos, com textos de 1 a 3 sentenças pequenas em português.

    # Classe Economia
    economia = [
        "A taxa de câmbio do dólar fechou em alta, atingindo R$5,20.",
        "O Banco Central aumentou a taxa Selic para 7,5%, visando conter a inflação.",
        "A bolsa de valores de São Paulo, a B3, fechou em queda de 1,3%.",
        "O PIB do Brasil cresceu 0,8% no último trimestre, superando as expectativas.",
    ]

    # Classe Esporte
    esporte = [
        "O Flamengo venceu o Corinthians por 2 a 0, liderando o Brasileirão.",
        "A seleção brasileira de vôlei conquistou a medalha de ouro nas Olimpíadas.",
        "Gabriel Medina é campeão mundial de surfe pela terceira vez.",
        "A maratonista brasileira quebrou o recorde sul-americano na Maratona de Berlim.",
    ]

    # Classe Ciências
    ciencias = [
        "Cientistas descobrem novo planeta semelhante à Terra em zona habitável.",
        "A vacina contra o novo coronavírus mostra eficácia de 95% em estudos.",
        "Pesquisadores brasileiros desenvolvem tecido que neutraliza o vírus da COVID-19.",
        "O telescópio espacial James Webb revela imagens inéditas de galáxias distantes.",
    ]

    # Gerando novos exemplos para o dataset textual, agora com um pouco mais de sentenças por exemplo.

    # Classe Economia
    economia_expandido = [
        "A inflação acumulada do ano atingiu 3,5%, segundo o IBGE. Especialistas apontam a alta nos preços dos alimentos como principal causa.",
        "Investimentos estrangeiros no Brasil crescem 15% no primeiro semestre, impulsionados pelo setor de tecnologia.",
        "O desemprego no Brasil cai para 11,8%, o menor índice em três anos. A retomada econômica pós-pandemia é vista como o principal fator.",
    ]

    # Classe Esporte
    esporte_expandido = [
        "O Brasil conquista sua primeira medalha de ouro no skate nas Olimpíadas, com destaque para a performance de Pedro Barros.",
        "A seleção feminina de futebol avança para a final da Copa do Mundo, após uma vitória emocionante contra a França por 2 a 1.",
        "Lewis Hamilton vence o Grande Prêmio do Brasil de Fórmula 1, marcando sua 100ª vitória na carreira.",
    ]

    # Classe Ciências
    ciencias_expandido = [
        "Nova terapia genética promete revolucionar o tratamento do Alzheimer, com primeiros testes mostrando redução significativa nos sintomas.",
        "A missão Mars Rover da NASA encontra evidências de água líquida sob a superfície de Marte, aumentando as esperanças de vida passada no planeta.",
        "Cientistas brasileiros desenvolvem novo material biodegradável a partir de resíduos da indústria de suco de laranja, visando reduzir o impacto ambiental.",
    ]

    # Adicionando os novos exemplos aos existentes
    economia = economia + economia_expandido
    esporte = esporte + esporte_expandido
    ciencias = ciencias + ciencias_expandido
    econom = [
        "The dollar exchange rate closed higher, reaching R$5.20.",
        "The dollar exchange rate closed higher, reaching R$5.20.",
        "The Central Bank raised the Selic rate to 7.5%, aiming to contain inflation.",
        "The São Paulo stock exchange, B3, closed down by 1.3%.",
        "The São Paulo stock exchange, B3, closed down by 1.3%.",
        "Brazil's GDP grew by 0.8% in the last quarter, exceeding expectations.",
        "Brazil's GDP grew by 0.8% in the last quarter, exceeding expectations.",
        "The year's accumulated inflation reached 3.5%, according to the IBGE. Experts point to the rise in food prices as the main cause.",
        "Foreign investments in Brazil grew by 15% in the first half, driven by the technology sector.",
        "Unemployment in Brazil falls to 11.8%, the lowest rate in three years. The post-pandemic economic recovery is seen as the main factor.",
    ]
    sports = [
        "Flamengo beat Corinthians 2-0, leading the Brasileirão.",
        "Flamengo beat Corinthians 2-0, leading the Brasileirão.",
        "The Brazilian volleyball team won the gold medal at the Olympics.",
        "Gabriel Medina is the world surfing champion for the third time.",
        "Gabriel Medina is the world surfing champion for the third time.",
        "The Brazilian marathon runner broke the South American record at the Berlin Marathon.",
        "The Brazilian marathon runner broke the South American record at the Berlin Marathon.",
        "Brazil wins its first gold medal in skateboarding at the Olympics, with Pedro Barros' performance a highlight.",
        "The women's soccer team advances to the World Cup final after an exciting 2-1 victory against France.",
        "Lewis Hamilton wins the Brazilian Grand Prix, marking his 100th career victory.",
    ]
    science = [
        "Scientists discover a new Earth-like planet in a habitable zone.",
        "Scientists discover a new Earth-like planet in a habitable zone.",
        "The vaccine against the new coronavirus shows 95% efficacy in studies.",
        "The vaccine against the new coronavirus shows 95% efficacy in studies.",
        "Brazilian researchers develop fabric that neutralizes the COVID-19 virus.",
        "The James Webb Space Telescope reveals unprecedented images of distant galaxies.",
        "The James Webb Space Telescope reveals unprecedented images of distant galaxies.",
        "New gene therapy promises to revolutionize Alzheimer's treatment, with initial tests showing significant symptom reduction.",
        "NASA's Mars Rover mission finds evidence of liquid water beneath the surface of Mars, increasing hopes for past life on the planet.",
        "Brazilian scientists develop a new biodegradable material from orange juice industry waste, aiming to reduce environmental impact.",
    ]

    # Consolidando os dados em um único dataset
    dataset = {
        # "texto": economia + esporte + ciencias,
        "text": econom + sports + science,
        "label_names": ["economic"] * len(econom)
        + ["sport"] * len(sports)
        + ["science"] * len(science),
    }
    mapper = {"economic": 0, "sport": 1, "science": 2}
    dataset = pd.DataFrame(dataset)
    dataset["label"] = [mapper[i] for i in dataset.label_names]
    print(dataset.sample(2))
    df_train, df_test = train_test_split(
        dataset, train_size=9 * 3, stratify=dataset.label
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    return (
        df_train.reset_index(drop=True),
        df_test.reset_index(drop=True),
        dataset.label_names.unique().tolist(),
        "toy",
    )

dic_datasets = {
    "ohsumed": get_ohsumed,
    "r8": get_r8,
    "r8_tiny": get_r8_tiny,
    "r52": get_r52,
    "mr": get_mr,
    "20news": get_20newsgroups,
    "cstr": get_cstr,
    "syskillwebert": get_syskillwebert,
    "dmozscience": get_dmozscience,
    "dmozhealth": get_dmozhealth,
    "classic4": get_classic4,
    "trec_fine": get_trec_fine,
    "trec": get_trec,
    "toy_data": get_toy_data,
    "webkb": get_webkb,
    "dmozcomputers": get_dmozcomputers,
    "dmozsports": get_dmozsports,
    "nsf": get_nsf,
    "snippets": get_snippets,
    "agnews": get_agnew,
    # get_imdb,
    # get_twitter,
}
    
def load_text_dataset(directory):
    data = []
    
    # Percorre as pastas que são as classes
    # import ipdb; ipdb.set_trace()
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        
        # Verifica se é uma pasta
        if os.path.isdir(label_path):
            # Percorre os arquivos dentro de cada pasta
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                # Abre e lê o conteúdo do arquivo
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        lines = file.readlines()
                        title = lines[0]
                        text = lines[1]
                        label = lines[-1]
                        # title = filename  # O título é o nome do arquivo
                        data.append([text, title, label])
                    except Exception as e:
                        print(file_path)
                        print(e)
        # Cria o DataFrame
    df = pd.DataFrame(data, columns=['text', 'title', 'label_names'])
    
    return df

def load_text_path(directory):
    data = []
    
    # Percorre as pastas que são as classes
    # import ipdb; ipdb.set_trace()
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        
        # Verifica se é uma pasta
        if os.path.isdir(label_path):
            label = label_path.split("/")[-1]
            # Percorre os arquivos dentro de cada pasta
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                # Abre e lê o conteúdo do arquivo
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        lines = file.readlines()
                        title = lines[0]
                        text = " ".join(lines)
                        # title = filename  # O título é o nome do arquivo
                        data.append([text, title, label])
                    except Exception as e:
                        print(file_path)
                        print(e)
    df = pd.DataFrame(data, columns=['text', 'title', 'label_names'])
    return df

def get_ohsumed_title():
    return get_ohsumed_root(True)

def get_ohsumed_root(only_title=False):
    df_train = load_text_path("datasets/ohsumed/training")
    df_test = load_text_path("datasets/ohsumed/test")
    
    le = preprocessing.LabelEncoder()
    le.fit(df_train["label_names"])
    df_train["label"] = le.transform(df_train["label_names"])
    if only_title:
        df_train["text"] = df_train["title"].apply(str)
        df_test["text"] = df_test["title"].apply(str)
        dataset_name = "ohsumed_root_title"
    else:
        df_train["text"] = df_train["text"].apply(str)
        df_test["text"] = df_test["text"].apply(str)
        dataset_name = "ohsumed_root"
        
    df_test["label"] = le.transform(df_test["label_names"])
    target_names = le.classes_
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    
    return df_train[columns_data+["title"]], df_test[columns_data+["title"]], target_names, dataset_name
    
import pandas as pd
import nltk
from nltk.corpus import twitter_samples

def get_twitter_10k():

    # Baixar os dados do Twitter se ainda não tiver baixado
    nltk.download('twitter_samples')

    # Carregar os dados
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    # Criação do DataFrame
    def create_dataframe(tweets, label, subset):
        return pd.DataFrame({
            'text': tweets,
            'label_names': label,
            'subset': subset
        })

    # Dividindo os dados em treino e teste
    train_pos_df = create_dataframe(positive_tweets[:5000], 'positive', 'train')
    test_pos_df = create_dataframe(positive_tweets[5000:], 'positive', 'test')

    train_neg_df = create_dataframe(negative_tweets[:5000], 'negative', 'train')
    test_neg_df = create_dataframe(negative_tweets[5000:], 'negative', 'test')

    # Concatenando todos os DataFrames
    df_train = pd.concat([train_pos_df, train_neg_df]).reset_index(drop=True)
    df_test = pd.concat([test_pos_df, test_neg_df]).reset_index(drop=True)
    
    le = preprocessing.LabelEncoder()
    le.fit(df_train["label_names"])
    df_train["label"] = le.transform(df_train["label_names"])
    df_test["label"] = le.transform(df_test["label_names"])
    target_names = le.classes_
    dataset_name = "twitter_10k"

    return df_train, df_test, target_names, dataset_name

def get_tag_my_news():

    # Exemplo de uso:
    directory_path = 'datasets/tag_my_news'
    data = load_text_dataset(directory_path)
    
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["label_names"])
    data["label"] = le.transform(data["label_names"])
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=TEST_SIZE, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "tag_my_news"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name

def get_mpqa():
    data = pd.read_csv("datasets/mpqa/mpqa.txt", header=None, sep="\t")
    data.columns = ["label_names", "text"]
    # total 16207
    le = preprocessing.LabelEncoder()
    le.fit(data["label_names"])
    data["label"] = le.transform(data["label_names"])
    data["text"] = data["text"].apply(str)
    df_train, df_test = train_test_split(
        data, test_size=TEST_SIZE, stratify=data.label, random_state=RANDOM_STATE
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_train["subset"] = "train"
    df_test["subset"] = "test"
    dataset_name = "mpqa"
    target_names = le.classes_
    return df_train[columns_data], df_test[columns_data], target_names, dataset_name

def get_split(df_full, config, target_names):
    if config.train_size_p * len(df_full) > len(target_names):
        df_train, df_val = train_test_split(df_full, train_size=config.train_size_p,
                                            random_state=config.exp_number,
                                            stratify=df_full.label)
    else:
        df_train, df_val = train_test_split(df_full, train_size=config.train_size_p,
                                            random_state=config.exp_number)
    return df_train, df_val

datasets = [
    get_ohsumed,
    get_r8,
    get_r8_tiny,
    get_r52,
    get_mr,
    get_20newsgroups,
    get_cstr,
    get_syskillwebert,
    get_dmozscience,
    get_dmozhealth,
    get_classic4,
    get_trec_fine,
    get_trec,
    get_toy_data,
    get_webkb,
    get_dmozcomputers,
    get_dmozsports,
    get_nsf,
    get_snippets,
    get_agnew,
    get_twitter_10k,
    get_ohsumed_title,
    get_ohsumed_root,
    get_trec_6,
    get_mpqa,
    get_tag_my_news,
    get_dblp,
    get_pge,
    # get_imdb,
    # get_twitter,
]

def best_max_lenght():
    return {
        "cstr": 288,
        "r8": 256,
        "r8_tiny": 256,
        "r52": 256,
        "ohsumed": 288,
        "movie_review": 64,
        "20newsgroups": 512,
        "agnews": 64,
        "snippets": 64,
        "tag_my_news": 64,
        "ohsumed_root": 288,
        "ohsumed_root_title": 32,
        "twitter_10k": 32,
        "TREC6": 32,
        "mpqa": 16,
        "dblp": 32,
        "pge": 398
    }
    
def setting_config(dataset, config, perc=0.01):
    if perc == 0.01:
        config = setting_config_001(dataset, config)
    if perc == 0.05:
        config = setting_config_005(dataset, config)
    if perc == 0.1:
        config = setting_config_010(dataset, config)
    if perc == 0.2:
        config = setting_config_020(dataset, config)
    return config

def setting_config_001(dataset, config):
    config["proj"] = "TC-Semisup-Setup-C"
    config["join_train_test"]=True
    config["compl_size"]=0
    config["val_size"]=1000
    if dataset == "ohsumed":
        config["train_size_per_class"]=False
        config["sampling_train"] = 207
        config["test_size"]=6193
    elif dataset == "r8":
        # 80 samples - 0.01%
        config["train_size_per_class"]=False
        config["sampling_train"] = 80
        config["test_size"]=6594
    elif dataset == "agnews":
        config["train_size_per_class"]=False
        config["sampling_train"] = 40
        config["test_size"]=6960
    elif dataset == "snippets":
        config["train_size_per_class"]=False
        config["sampling_train"] = 64
        config["test_size"]=11276
    elif dataset == "dblp":
        config["train_size_per_class"]=False
        config["sampling_train"] = 120
        config["test_size"]=22880
    elif dataset == "20newsgroups":
        config["train_size_per_class"]=True
        config["sampling_train"] = 10
        config["test_size"]=17646
    elif dataset == "TREC6":
        config["train_size_per_class"]=True
        config["sampling_train"] = 10
        config["test_size"]=4940
    return config

def setting_config_005(dataset, config):
    config["proj"] = "TC-Semisup-Setup-C"
    config["join_train_test"]=True
    config["compl_size"]=0
    config["val_size"]=1000
    if dataset == "ohsumed":
        config["train_size_per_class"]=False
        config["sampling_train"] = 370
        config["test_size"]=6030
    elif dataset == "r8":
        config["train_size_per_class"]=False
        config["sampling_train"] = 383
        config["test_size"]=6291
    elif dataset == "snippets":
        config["train_size_per_class"]=False
        config["sampling_train"] = 617
        config["test_size"]=10723
    elif dataset == "dblp":
        config["train_size_per_class"]=False
        config["sampling_train"] = 1200
        config["test_size"]=21800
    elif dataset == "TREC6":
        config["train_size_per_class"]=False
        config["sampling_train"] = 300
        config["test_size"]=4700
    elif dataset == "agnews":
        config["train_size_per_class"]=False
        config["sampling_train"] = 400
        config["test_size"]=6600
    elif dataset == "20newsgroups":
        assert 1 == 0, "dataset not config"
        # config["train_size_per_class"]=True
        # config["sampling_train"] = 10
        # config["test_size"]=17646
    return config

def setting_config_020(dataset, config):
    config["proj"] = "TC-Semisup-Setup-C"
    config["join_train_test"]=True
    config["compl_size"]=0
    config["val_size"]=1000
    if dataset == "ohsumed":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 1480
        config["test_size"]=4920
    elif dataset == "r8":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 1534
        config["test_size"]=5139
    elif dataset == "snippets":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 2468
        config["test_size"]=8872
    elif dataset == "TREC6":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 1200
        config["test_size"]=3800
    elif dataset == "dblp":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 4800
        config["test_size"]=18200
    elif dataset == "agnews":
        # 0.2%
        config["train_size_per_class"]=False
        config["sampling_train"] = 1600
        config["test_size"]=5400
    elif dataset == "20newsgroups":
        config["train_size_per_class"]=True
        config["sampling_train"] = 10
        config["test_size"]=17646
    return config


def setting_config_010(dataset, config):
    config["proj"] = "TC-Semisup-Setup-C"
    config["join_train_test"]=True
    config["compl_size"]=0
    config["val_size"]=1000
    if dataset == "ohsumed":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 740
        config["test_size"]=5660
    elif dataset == "r8":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 767
        config["test_size"]=5907
    elif dataset == "agnews":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 800
        config["test_size"]=6200
    elif dataset == "snippets":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 1234
        config["test_size"]=10106
    elif dataset == "dblp":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 2400
        config["test_size"]=20600
    elif dataset == "20newsgroups":
        config["train_size_per_class"]=True
        config["sampling_train"] = 10
        config["test_size"]=17646
    elif dataset == "TREC6":
        # 0.1%
        config["train_size_per_class"]=False
        config["sampling_train"] = 600
        config["test_size"]=4400
    return config

def choose_setup(name, dataset, perc=0.01):
    config = {}
    if name == "A":
        config["proj"] = "TC-Semisup-Setup-A"
        config["join_train_test"]=True
        config["train_size_per_class"]=True
        config["sampling_train"] = 20
        config["compl_size"]=1000
        if dataset == "tag_my_news":
            config["val_size"]=140
            config["test_size"]=31269
        elif dataset == "snippets":
            config["val_size"]=160
            config["test_size"]=11020
        elif dataset == "ohsumed_root_title":
            config["val_size"]=419
            config["test_size"]=5502
        elif dataset == "movie_review":
            config["val_size"]=40
            config["test_size"]=3554
        elif dataset == "twitter_10k":
            config["val_size"]=40
            config["test_size"]=8920
    elif name == "C":
        config = setting_config(dataset, config, perc=perc)
    elif name == "B":
        config["proj"] = "TC-Semisup-Setup-B"
        config["join_train_test"]=True
        config["compl_size"]=0
        config["val_size"]=1000
        if dataset == "mpqa":
            config["train_size_per_class"]=True
            config["sampling_train"] = 10
            config["test_size"]=9583
        elif dataset == "agnews":
            config["train_size_per_class"]=True
            config["sampling_train"] = 10
            config["test_size"]=6960
        elif dataset == "TREC6":
            config["train_size_per_class"]=True
            config["sampling_train"] = 10
            config["test_size"]=4940
        
    return config

def build_stats(datasets=datasets):
    stats = []
    for func in tqdm(datasets):
        stats.append(get_stats(*func()))
        
    datasets_stats = pd.DataFrame(stats)
    format_2_decimals = lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x
    # Aplicar a função a todas as colunas numéricas
    datasets_stats = datasets_stats.map(format_2_decimals)
    datasets_stats.to_csv("artifacts/docs/dataset_stats.csv", index=False)
    # datasets_stats.to_excel("artifacts/docs/dataset_stats.xlsx", index=False)
    datasets_stats.to_markdown("artifacts/docs/dataset_stats.md", index=False)
    datasets_stats.to_html("artifacts/docs/dataset_stats.html", index=False)

# build_stats()