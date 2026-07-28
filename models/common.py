import datetime
import json
import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import datasets
import nltk
from utils.utils import send_msg
import logging

# Função para adicionar ruído gaussiano
def add_gaussian_noise( embeddings, noise_level=0.01):
    noise = torch.normal(0, noise_level, size=embeddings.size()).to(embeddings.device)
    noisy_embeddings = embeddings + noise
    return noisy_embeddings
    
def get_dataset_splited(func, exp_number, settings, ):
    
    train_size=0.9
    
    if "sampling_train" in settings:
        train_size = settings["sampling_train"]
    
    if isinstance(func, datasets.dataset_dict.DatasetDict):
        df_train = pd.DataFrame({"text": list(func['train']["text"]), "label": list(func['train']["label"])})
        df_val = pd.DataFrame({"text": func['validation']["text"], "label": func['validation']["label"]})
        df_test = pd.DataFrame({"text": func['test']["text"], "label": func['test']["label"]})
        dataset_name = "scotus"
        target_names = list(range(df_train.label.nunique()))
        if "sampling_train" in settings:
            df_train, df_val = train_test_split(
                df_train, train_size=train_size, stratify=df_train.label, random_state=exp_number
            )
            df_full = pd.concat([df_train, df_val])
        else:
            train_size = len(df_train) / (len(df_train) + len(df_val))
    else:
        df_full, df_test, target_names, dataset_name = func()
        # print(df_full.head(20))
        df_train, df_val = train_test_split(
            df_full, train_size=train_size, stratify=df_full.label, random_state=exp_number
        )
        if "sampling_train" in settings:
            df_full = pd.concat([df_train, df_val])
            
    return df_full, df_train, df_val, df_test, train_size, target_names, dataset_name
            
from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(data, max_train_per_class, target_column='label',  min_test_per_class=2,
                     random_state=None):
    X_train, X_test = [], []
    
    data = data.reset_index(drop=True)
    rng = pd.Series(data.index).sample(frac=1, random_state=random_state)  # Shuffle indices with random_state
    data = data.loc[rng.index]  # Shuffle data
    
    # Iterate over each class
    for class_label in data[target_column].unique():
        # Filter data for the current class
        class_data = data[data[target_column] == class_label]
        
        # Ensure there are enough examples to meet the test set requirement
        if len(class_data) > max_train_per_class + min_test_per_class:
            train_size = min(max_train_per_class, len(class_data) - min_test_per_class)
            test_size = len(class_data) - train_size
            train_data, test_data = train_test_split(
                class_data, test_size=test_size, train_size=train_size, stratify=None,
                random_state=random_state
            )
        else:
            # Allocate minimum examples for the test set and the rest for training
            test_data = class_data.sample(n=min_test_per_class,
                                          random_state=random_state)
            train_data = class_data.drop(test_data.index)
        
        # Append to the train and test sets
        X_train.append(train_data)
        X_test.append(test_data)
    
    # Concatenate the lists back into DataFrames/arrays
    X_train = pd.concat(X_train)
    X_test = pd.concat(X_test)
    
    return X_train, X_test


def stratified_split_by_class_df(df, target_column, n, random_state=None):
    df_train_list, df_test_list = [], []
    unique_classes = df[target_column].unique()
    
    for cls in unique_classes:
        # Seleciona todos os exemplos da classe atual
        df_cls = df[df[target_column] == cls]
        
        # Divide a classe em treino e teste
        df_cls_train, df_cls_test = train_test_split(
            df_cls, train_size=n, random_state=random_state, shuffle=True)
        
        # Adiciona os exemplos ao conjunto de treino e teste
        df_train_list.append(df_cls_train)
        df_test_list.append(df_cls_test)
    
    # Concatena todos os exemplos
    df_train = pd.concat(df_train_list).reset_index(drop=True)
    df_test = pd.concat(df_test_list).reset_index(drop=True)
    
    return df_train, df_test

from collections import Counter
import random
def sample_ids(ids, labels, n=10, seed=None):
    """
    Retorna uma lista de IDs balanceada com n exemplos de cada classe.
    Se faltar exemplos, completa com a classe mais frequente até atingir
    n * quantidade de classes. Também retorna os IDs que sobraram.

    Args:
        ids (list): Lista de IDs dos exemplos.
        labels (list): Lista de rótulos correspondentes aos IDs.
        n (int): Número de exemplos por classe.
        seed (int, optional): Semente para controlar a aleatoriedade.

    Returns:
        tuple: (lista_balanceada, lista_restante)
    """
    if seed is not None:
        random.seed(seed)
    
    # Agrupar IDs por classe
    grouped = {}
    for id_, label in zip(ids, labels):
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(id_)

    # Selecionar n exemplos de cada classe
    balanced_ids = []
    for label, id_list in grouped.items():
        balanced_ids.extend(random.sample(id_list, min(n, len(id_list))))
    
    # Contar quantos exemplos faltam
    total_classes = len(grouped)
    missing_count = total_classes * n - len(balanced_ids)

    # Preencher com IDs da classe mais frequente
    if missing_count > 0:
        most_frequent_label = Counter(labels).most_common(1)[0][0]
        balanced_ids.extend(random.choices(grouped[most_frequent_label], k=missing_count))
        
    # Identificar IDs restantes
    used_ids = set(balanced_ids)
    remaining_ids = [id_ for id_ in ids if id_ not in used_ids]

    return balanced_ids, remaining_ids

def detailed_split(func, exp_number, train_size, sample_per_class=10, val_size=1000, test_size=6960, compl_size=0, 
                   join_train_test=False, train_size_per_class=True):
    
    df_full, df_test, target_names, dataset_name = func()
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    
    dic_data = {
        "ohsumed": {"cl": 9,"max": 207, "val": 1000, "te": 6193},
        "r8": {"cl": 10,"max": 80, "val": 1000, "te": 6594},
        "agnews": {"cl": 10, "max": 40, "val": 1000, "te": 6960},
        "snippets": {"cl": 8, "max": 64, "val": 1000, "te": 11276},
        "dblp": {"cl": 20, "max": 120, "val": 1000, "te": 22880},
        # "ohsumed": {"tr": 207, "val": 1000, "te": 6193},
        # "ohsumed": {"tr": 207, "val": 1000, "te": 6193},
    }
    samples = dic_data[dataset_name]
    # sp_val = samples["val"]
    # sp_te = samples["te"]
    sp_tr = samples["cl"]
    
    train_ids = df_full.index
    labels = df_full.label
    train_ids, remain = sample_ids(train_ids, labels, n=sp_tr, seed=exp_number)
    
    df_train = df_full.loc[train_ids].copy()
    df_remain = df_full.loc[remain].copy()
    if train_size < len(df_train):
        diff = len(df_train)-train_size
        sampled = df_train.sample(diff, random_state=exp_number)
        df_remain = pd.concat([df_remain, sampled.copy()])
        df_train = df_train.drop(sampled.index)
    
    elif train_size > len(df_train):
        diff = train_size-len(df_train)
        sampled = df_remain.sample(diff, random_state=exp_number)
        df_train = pd.concat([df_train, sampled.copy()])
        df_remain = df_remain.drop(sampled.index)
        
    if val_size < len(df_remain):
        sample = df_remain.sample(val_size, random_state=exp_number)
        df_val = sample.copy()
        df_remain = df_remain.drop(sample.index)
        df_test = pd.concat([df_test, df_remain])
    else:
        df_val = df_remain.copy()
        
    if test_size < len(df_test):
        diff = len(df_test)-test_size
        sampled = df_test.sample(diff, random_state=exp_number)
        df_test = df_test.drop(sampled.index)
    df_compl = pd.DataFrame()
    
    # if join_train_test:
    #     df_full = pd.concat([df_full, df_test]).reset_index(drop=True)
    #     df_full, df_test = train_test_split(df_full, test_size=test_size, stratify=df_full.label, random_state=exp_number)
    # else:
    #     df_test, _ = train_test_split(df_test, train_size=test_size, stratify=df_test.label, random_state=exp_number)    
    
    # if train_size_per_class:
    #     if compl_size > 0:
    #         df_train, df_val = stratified_split(df_full, max_train_per_class=train_size, target_column='label',  
    #                                             min_test_per_class=2, random_state=exp_number)
    #     else:
    #         df_train, df_val = stratified_split(df_full, max_train_per_class=train_size, target_column='label',  
    #                                             min_test_per_class=1, random_state=exp_number)
    # else:
    #     df_train, df_val = train_test_split(df_test, train_size=train_size, stratify=df_test.label, random_state=exp_number)    
        
    # if len(df_val) > val_size:
    #     df_val, df_rest = train_test_split(df_val, train_size=val_size, stratify=df_val.label, random_state=exp_number)
        
    # if compl_size > 0: 
    #     if compl_size == len(df_rest):
    #         df_compl = df_rest
    #     else:
    #         df_compl, _ = train_test_split(df_rest, train_size=compl_size, random_state=exp_number)     
    # else:
    #     df_compl = pd.DataFrame()
    
    return df_full, df_train, df_compl, df_val, df_test, train_size, target_names, dataset_name

def generate_topics(df, model_name, n=3, diversity=0.7):
    from keybert import KeyBERT
    
    kw_model = KeyBERT(model=model_name)
    docs = df.text.to_list()
    doc_embeddings, word_embeddings = kw_model.extract_embeddings(docs, min_df=1, stop_words="english",
                                                                  keyphrase_ngram_range=(1, 3), 
                                                                  )
    doc_keywords = kw_model.extract_keywords(docs, min_df=1, stop_words="english", 
                                        doc_embeddings=doc_embeddings, 
                                        word_embeddings=word_embeddings,
                                        keyphrase_ngram_range=(1, 3), 
                                        top_n=10,
                                        diversity=diversity
                                        )
    topics = [] 
    count_label = {}
    for keywords, cl in zip(doc_keywords, df.label):
        if cl not in count_label: count_label[cl] = 0
        if count_label[cl] < 6:
            topics.extend([[tops[0], cl] for tops in keywords[:8]])
            count_label[cl] += 1
    df = pd.DataFrame(topics, columns=["text_aug", "label"])
    
    # import ipdb; ipdb.set_trace()
    return df

def generate_topics_lda():
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.models import Phrases
    import gensim
    from gensim import corpora
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    
    stop_words = stopwords.words('english')
    # Converter os documentos para formato necessário para LDA
    texts = [[word for word in doc.lower().split() if word not in stop_words] for doc in df['text']]
    # Gerar bigrams e trigrams usando gensim
    bigram = Phrases(texts, min_count=5, threshold=100)  # Define os bigrams
    trigram = Phrases(bigram[texts], threshold=100)  # Define os trigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Função para aplicar bigrams e trigrams aos textos
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Aplicar bigrams e trigrams aos textos
    texts_bigrams = make_bigrams(texts)
    texts_trigrams = make_trigrams(texts_bigrams)

    # Criar dicionário e corpus para LDA
    dictionary = corpora.Dictionary(texts_trigrams)
    corpus = [dictionary.doc2bow(text) for text in texts_trigrams]

    # corpus = df.text.to_list()
    # Modelagem de tópicos usando LDA
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, random_state=42, passes=10)

    # Visualizar os tópicos gerados
    topics = lda_model.show_topics(num_words=20, formatted=False)
    return topics

def get_n_samples(df, n):
    return df.groupby('label').apply(lambda x: x.sample(min(n, len(x)))).reset_index(drop=True)


def generate_aug(df, num_gen=2, repeats=1, model_name=None, augs=[]):
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    import nlpaug.flow as naf
    from nlpaug.util import Action
    import nltk
    
    nltk.download('averaged_perceptron_tagger_eng')
    
    data_augs = []
    # import ipdb; ipdb.set_trace()
    
    if "keyaug" in augs:        
        aug_key = nac.KeyboardAug()
        exp_key = []
        
        for i in range(repeats):
            base_cru_1p_keyaug = df.copy()
            base_cru_1p_keyaug['text_aug'] = base_cru_1p_keyaug['text'].apply(lambda x:aug_key.augment(x,num_gen))
            base_cru_1p_keyaug = base_cru_1p_keyaug.explode('text_aug').reset_index(drop=True)
            # base_cru_1p_keyaug = get_n_samples(base_cru_1p_keyaug, num_gen)
            exp_key.append(base_cru_1p_keyaug)
        
        data_augs.append(pd.concat(exp_key))
        
    if "eda" in augs:
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        import nlpaug.flow as naf

        aug_eda = naf.Sequential([
            naf.Sometimes([naw.RandomWordAug(action="swap")]),
            naf.Sometimes([naw.RandomWordAug(action="delete")]),
            naf.Sometimes([naw.SynonymAug(aug_src='wordnet')])
        ])
        exp_eda = []

        for i in range(repeats):
            base_cru_1p_eda = df.copy()
            base_cru_1p_eda['text_aug'] = base_cru_1p_eda['text'].apply(lambda x:aug_eda.augment(x,num_gen))
            base_cru_1p_eda = base_cru_1p_eda.explode('text_aug').reset_index(drop=True)
            # base_cru_1p_eda = get_n_samples(base_cru_1p_eda, num_gen)
            exp_eda.append(base_cru_1p_eda)
    
        data_augs.append(pd.concat(exp_eda))
        
    if "topics" in augs:
        df_topics = generate_topics(df, model_name, n=num_gen, diversity=0.7)
        data_augs.append(df_topics)
        
    df_ = pd.concat(data_augs).reset_index(drop=True)
    return df_
    

def get_dataset_splited_semisup(func, exp_number, settings, ):
    
    train_size=0.9
    
    if "sampling_train" in settings:
        train_size = settings["sampling_train"]
    
    if isinstance(func, datasets.dataset_dict.DatasetDict):
        df_train = pd.DataFrame({"text": list(func['train']["text"]), "label": list(func['train']["label"])})
        df_val = pd.DataFrame({"text": func['validation']["text"], "label": func['validation']["label"]})
        df_test = pd.DataFrame({"text": func['test']["text"], "label": func['test']["label"]})
        dataset_name = "scotus"
        target_names = list(range(df_train.label.nunique()))
        if "sampling_train" in settings:
            df_train, df_val = train_test_split(
                df_train, train_size=train_size, stratify=df_train.label, random_state=exp_number
            )
            df_full = pd.concat([df_train, df_val])
        else:
            train_size = len(df_train) / (len(df_train) + len(df_val))
    else:
        df_full, df_test, target_names, dataset_name = func()
        # print(df_full.head(20))
        if train_size < 1:
            df_train, df_val = train_test_split(
                df_full, train_size=train_size, stratify=df_full.label, random_state=exp_number
            )
        else:
            # df_train, df_val = stratified_split_by_class_df(df_full, 'label', 
            #                                                 train_size, random_state=exp_number)
            df_train, df_val = stratified_split(df_full, max_train_per_class=train_size, target_column='label',  
                                                min_test_per_class=2, random_state=exp_number)
            
        
        df_compl, df_val = train_test_split(
            df_val, train_size=0.8, stratify=df_val.label, random_state=exp_number
        )
        if "sampling_train" in settings:
            if train_size < 1:
                df_full = pd.concat([df_train, df_val])
            else:
                df_full = pd.concat([df_train, df_compl, df_val])
            
    return df_full, df_train, df_compl, df_val, df_test, train_size, target_names, dataset_name

def get_configs():
    caminho_do_arquivo = "config.json"
    if not os.path.exists(caminho_do_arquivo):
        return {"machine": "local", "url": None}
    with open(caminho_do_arquivo, "r") as arquivo:
        return json.load(arquivo)


def call_exp(funcs_datasets, random_states, run_experiment, name_method, notify=True, settings={}):
    config = get_configs()
    machine = config["machine"]
    url = config["url"]
        
    for func in funcs_datasets:
        results = []
        init = datetime.datetime.now()
        
        if hasattr(func, "__name__"):
            name = func.__name__
        else:
            name = "scotus"
        if notify:
            send_msg(
                f"Starting...\nexperiment: {name_method} \ndataset: {name}",
                user=machine,
                url=url,
            )
        for i in random_states:
            res = run_experiment(i, func, settings)
            name_method = res["class_name"]
            results.append(res)
            if notify:
                if "wandb_id" in res:
                    msg = (f"experiment: {name_method}_{i} \n"
                           f"dataset: {name} acc: {res['acc']}\n"
                           f"wandb_id: {res['wandb_id']}")
                else:
                    msg = f"experiment: {name_method}_{i} \ndataset: {name} acc: {res['acc']}"
                send_msg(
                    msg,
                    user=machine,
                    url=url,
                )

        current_day_hour = datetime.datetime.now()
        total = current_day_hour - init
        current_day_hour = current_day_hour.strftime("%Y-%m-%d_%H-%M")
        df = pd.DataFrame(results)
        name_file = f"artifacts/results/{name}_{name_method}_{current_day_hour}"
        df.to_pickle(f"{name_file}.pkl")
        if "pred" in df.columns:
            df.drop(columns=["pred"], inplace=True)
        if "prediction" in df.columns:
            df.drop(columns=["prediction"], inplace=True)
        df.to_csv(f"{name_file}.csv", index=False)

        wandb_sync = "None"
        if "wandb_sync" in df.columns:
            wandb_sync = "\n".join(df.wandb_sync.unique())
        if True:
            send_msg(
                (
                    f"Summary: {name} \nmethod: {name_method} acc mean: {df.acc.mean()}"
                    f" machine: {machine} time: {total}\n"
                    f"WANDB SYNC:\n{wandb_sync}"
                ),
                user=machine,
            )
            # for i in range(len(df)):
            #     send_msg(
            #         f"{df.loc[i].to_dict()},",
            #         user=machine,
            #     )

    return (
        f"Method: {name_method} machine: {machine} time: {total}"
    )

def create_stratified_batches(df, batch_size):
    # Separando o DataFrame por classes
    classes = df["label"].unique()
    class_groups = [df[df["label"] == c].copy() for c in classes]

    # Criando os batches
    batches = []
    batch = pd.DataFrame()
    while any(len(group) > 0 for group in class_groups):
        for group in class_groups:
            if len(group) > 0:
                # Tentando manter o batch balanceado entre as classes
                sample_size = min(len(group), max(1, batch_size // len(classes)))
                sample = group.sample(sample_size, replace=False)
                batch = pd.concat([batch, sample])
                group.drop(sample.index, inplace=True)

                # Se o batch alcançou o tamanho desejado, salva e começa um novo
                if len(batch) >= batch_size:
                    batches.append(batch)
                    batch = pd.DataFrame()  # Resetando o batch para o próximo

    # Adicionando o último batch se ele contiver dados e não atingiu o tamanho total
    if not batch.empty:
        batches.append(batch)

    return batches


def save_embeddings(embeddings, filename):
    """
    Salva embeddings em disco no formato numpy (.npy).

    Args:
    embeddings (np.ndarray or torch.Tensor): Embeddings para serem salvos.
    filename (str): Nome do arquivo para salvar os embeddings.

    Returns:
    None
    """
    # Verifica se os embeddings são uma instância de tensor do PyTorch
    if isinstance(embeddings, torch.Tensor):
        # Converte para numpy
        embeddings = embeddings.numpy()
    elif isinstance(embeddings, list):
        # Converte para numpy
        embeddings = np.array(embeddings)
    elif not isinstance(embeddings, np.ndarray):
        raise ValueError("O formato de entrada deve ser np.ndarray ou torch.Tensor")

    # Salva o array numpy no disco
    np.save(filename, embeddings)
    
class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return idx, text, label
        return idx, text, None

