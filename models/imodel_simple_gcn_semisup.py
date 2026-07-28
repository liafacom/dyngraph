import pandas as pd
import gc
import logging
from models.basemodel import AttrDict, BaseModel
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from BertGNN.data2graph import Data2Graph, FAISSGraphBuilder, calculate_heterophily_edges
from models.common import add_gaussian_noise, detailed_split, call_exp, generate_aug, get_dataset_splited, get_dataset_splited_semisup, save_embeddings, TextDataset
from utils.utils import (
    best_max_lenght,
    check_log_path,
    choose_setup,
    get_20newsgroups,
    get_bbcsport,
    get_cstr,
    get_dblp,
    get_dmozscience,
    get_mr,
    get_ohsumed,
    get_ohsumed_root,
    get_ohsumed_title,
    get_r8,
    get_r8_double,
    get_r52,
    get_snippets,
    get_syskillwebert,
    get_tag_my_news,
    get_toy_data,
    get_trec,
    get_agnew,
    get_twitter_10k,
    get_trec_6,
    get_mpqa,
    send_msg,
)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # token_embeddings = model_output[
    #     0
    # ]  # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def normalize(model_output, encoded_input=None, attention_mask=None, type_emb="cls_norm"):
    if "cls" in type_emb :
        embeddings_bert = model_output.last_hidden_state[
            :, 0, :
        ]  # Pega o embedding do token [CLS]
        if "norm" in type_emb:
            embeddings_bert = F.normalize(embeddings_bert, p=2, dim=1)
        return embeddings_bert
    if attention_mask is None and encoded_input is not None:
        attention_mask = encoded_input["attention_mask"]

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    if "norm" in type_emb:
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


class MulticlassClassificationHead(nn.Module):
    """Uma camada de classificação para tarefas de classificação multiclasse."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(MulticlassClassificationHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)
        self.hidden_dim = hidden_dim
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)

        return output


class ModelGCN(nn.Module, BaseModel):
    def __init__(
        self,
        config,
        hidden_dim=512,
        k=3,
        m=0.7,
        mode="connectivity",
        dropout=0.6,
        verbose=0,
    ):
        super(ModelGCN, self).__init__()
        self.name_method = "model_gat_simple"
        self.dropout = dropout
        self.mode = mode
        self.m = m
        self.verbose = verbose
        self.df_compl = None
        self.test_every_epoch = True
    
    def setting(
        self,
        config
    ):
        self.config = AttrDict(config)
        self.results = AttrDict()         
        self.auto_model = AutoModel.from_pretrained(self.config.transformer_name,
                                               output_attentions=True,
                                                output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_name)
        self.numero_de_classes = self.config.n_classes
        self.name_model = self.config.name_model
        self.dropout = self.config.dropout_
        self.mode = self.config.mode
        self.m = self.config.m
        hidden_dim = self.config.hidden_dim
        input_dim = self.auto_model.config.hidden_size
        n_layers = self.config.n_layers
        self.max_length = self.config.max_length
        self.emb_bert = self.config.emb_bert
        self.emb_gnn = self.config.emb_gnn
                
        self.classification_head = MulticlassClassificationHead(
            input_dim, hidden_dim=hidden_dim, output_dim=self.numero_de_classes
        )
        if self.config.layer_norm:
            self.norms = nn.ModuleList()
            self.norms.append(nn.LayerNorm(input_dim))
            for n in range(n_layers-1):
                self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.gnns = nn.ModuleList()
            
        if "heads" not in config:
            self.gnns.append(GCNConv(input_dim, hidden_dim, bias=True))        
            for n in range(n_layers-1):
                self.gnns.append(GCNConv(hidden_dim, hidden_dim, bias=True))
            self.gnns.append(GCNConv(hidden_dim, self.numero_de_classes, bias=False))
        else:
            heads = self.config.heads        
            self.gnns.append(GATv2Conv(input_dim, hidden_dim, heads=heads[0], bias=True, 
                                    #    dropout=self.dropout
                                       ))        
            for n in range(n_layers-1):
                self.gnns.append(GATv2Conv(heads[n]*hidden_dim, hidden_dim, heads=heads[n+1], bias=True, 
                                        #    dropout=self.dropout
                                           ))
            self.gnns.append(GATv2Conv(heads[-2] * hidden_dim, self.numero_de_classes, 
                                    heads=heads[-1], bias=False, #dropout=self.dropout
                                    ))
        
        
        if self.config.lr_lambda > 0:
            if self.config.kind_lambda == "concat":
                self.linear_loss = nn.Linear(2*self.numero_de_classes, 1)
            else:
                self.linear_loss = nn.Linear(self.numero_de_classes, 1)
            self.linear_loss.reset_parameters()
            
        for n in range(n_layers):
            self.gnns[n].reset_parameters()
        
        if config['freeze_bert']:
            for param in self.auto_model.parameters():
                param.requires_grad = False
    
    def cosine_similarity(self, x1, x2):
        return nn.functional.cosine_similarity(x1, x2)
    
    def contrastive_loss(self, batch_embeddings, batch_labels, margin=1.0):
        batch_size = batch_embeddings.size(0)
        loss = 0.0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                distance = self.cosine_similarity(batch_embeddings[i].unsqueeze(0), batch_embeddings[j].unsqueeze(0))
                label = 1 if batch_labels[i] == batch_labels[j] else 0
                loss += (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
        loss /= (batch_size * (batch_size - 1)) / 2  # Média sobre os pares
        return loss
    
    def forward(self, input_ids, encoded_input, idx, token_type_ids=None, attention_mask=None):
        # Inputs -> BERT -> outputs
        outputs = self.auto_model(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        
        # Embeddings for GNN
        embeddings_gnn = normalize(outputs, encoded_input, type_emb=self.emb_gnn)
        # Embeddings for Bert
        embeddings_bert = normalize(outputs, encoded_input, type_emb=self.emb_bert)
        
        # noise component
        if self.training:
            embeddings_gnn = add_gaussian_noise(embeddings_gnn)
            if "cls" in self.emb_bert:
                embeddings_bert = add_gaussian_noise(embeddings_bert)
            else:
                embeddings_bert = embeddings_gnn
            
        data = self.fg.add_points(embeddings_gnn)
        data = data.to(self.device)
        edge_index = data.edge_index
        edge_weight = None
        if self.config.edge_weight:
            edge_weight = data.edge_attr
        bsize = len(embeddings_bert)
        cls_logit = self.classification_head(embeddings_bert)
        cls_pred = nn.Softmax(dim=1)(cls_logit)
        x = data.x
        x[-bsize:, :] = embeddings_gnn
        h = x
        if self.mode == "connectivity" or edge_weight is None:
            for l in range(self.config.n_layers-1):
                if self.config.layer_norm:
                    h = self.norms[l](h)
                if l != 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                h = self.gnns[l](h, edge_index)
                h = F.elu(h)
            h = self.gnns[-1](h, edge_index)
        else:
            for l in range(self.config.n_layers-1):
                if self.config.layer_norm:
                    h = self.norms[l](h)
                if l != 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                h = self.gnns[l](h, edge_index, edge_weight=edge_weight)
                h = F.elu(h)
            h = self.gnns[-1](h, edge_index, edge_weight=edge_weight)

        gcn_pred = nn.Softmax(dim=1)(h)
        if "lr_lambda" in self.config and self.config.lr_lambda > 0:
            if self.config.kind_lambda == "minus":
                loss_input = h[-bsize:,:] - cls_logit
            if self.config.kind_lambda == "plus":
                loss_input = h[-bsize:,:] + cls_logit
            if self.config.kind_lambda == "concat":
                loss_input = torch.concat((h[-bsize:,:], cls_logit), dim=1)
            if self.config.kind_lambda == "minus2":
                loss_input = cls_logit - h[-bsize:,:]
            lambda_opt = F.sigmoid(self.linear_loss(loss_input))
            pred = (gcn_pred[-bsize:, :] + 1e-10) * lambda_opt + cls_pred * (1 - lambda_opt)
        else:
            lambda_opt = torch.tensor([-1])
            pred = (gcn_pred[-bsize:, :] + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)

        return pred, embeddings_bert, F.log_softmax(pred, dim=1), lambda_opt
    
    def setup_model(self):
        self = self.to(self.config.device)
        self.fg = FAISSGraphBuilder(k=self.config.k, similar_min=self.config.similar_min,
                                    flag_only_class=self.config.restrictive_class)
        
        list_gcns = [
                {"params": self.auto_model.parameters(), "lr": self.config.lr_distilbert},
                {"params": self.classification_head.parameters(), "lr": self.config.lr_distilbert}
            ]
        if "lr_lambda" in self.config and self.config.lr_lambda > 0:
            list_gcns.append({"params": self.linear_loss.parameters(), "lr": self.config.lr_lambda})
        for n in range(self.config.n_layers):
            list_gcns.append({"params": self.gnns[n].parameters(), "lr": self.config.lr_gnn})
        
        if self.config.optimizer == "AdamWScheduleFree":
            from schedulefree import AdamWScheduleFree
            self.optimizer = AdamWScheduleFree(list_gcns)
        else:
            self.optimizer = optim.AdamW(list_gcns)
        
        num_training_steps = self.config.num_epochs * self.config.train_size
        num_warmup_steps = int(0.1 * num_training_steps)
        logging.info(f"Warmup steps: {num_warmup_steps}")
        logging.info(f"Train steps: {num_training_steps}")

        self.lr_scheduler = get_scheduler(
            name="linear",  # Outros exemplos incluem "cosine", "cosine_with_restarts"
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,  # Número de passos de warmup
            num_training_steps=num_training_steps,
        )
        self.loss_func = F.nll_loss
        
    def plot_lambda(self, lambda_opts):
        lambda_opts = np.concatenate(lambda_opts)
        lambda_opts = np.round(lambda_opts, 2)
        plt.hist(lambda_opts, bins=20, edgecolor='black')
        plt.title('Histograma do Lambda entre loss bert e gnn')
        plt.xlabel('lambda')
        plt.ylabel('Frequência')
        # plt.grid(True)
        plt.savefig(f'artifacts/imgs/{self.config.dataset_name}/histograma_{self.mode_eval}_{self.epoch}.png')  # Salva o gráfico como um arquivo PNG
        plt.close()
        
    def run_epoch(self, epoch):
        total_loss = 0
        step = 0
        
        # join train + aug data
        self.df = pd.concat([self.df_train, self.df_compl, self.df_test])
        self.df = self.df.reset_index(drop=True)
        feats_local = self.get_train_embeddings(self.df)
        gc.collect()
        torch.cuda.empty_cache()
        # building graph with df pandas
        self.fg.build_graph_data(feats_local, self.df.label.values)
        
        # with labeled data aug
        # dataset = TextDataset(self.df.text.values, self.df.label.values)
        # without labeled data aug
        df_training  = pd.concat([self.df_train, self.df_compl])
        df_training = df_training.reset_index(drop=True)
        dataset = TextDataset(df_training.text, df_training.label)
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        lambda_opts = []
        self.train()
        for idx, texts, labels in tqdm(dataloader):
            step += 1
            # Tokenize sentences
            encoded_input = self.tokenizer(
                list(texts), 
                return_tensors='pt',          # Retorna tensores PyTorch
                padding='max_length',         # Preenche até o comprimento máximo
                truncation=True,              # Trunca se o texto for muito longo
                max_length=self.max_length,               # Define o comprimento máximo de tokens
                return_attention_mask=True,   # Retorna a máscara de atenção
                return_token_type_ids=False,  # Não retorna IDs de tipo de token
                add_special_tokens=True       # Adiciona tokens especiais [CLS] e [SEP]
            )
            bsize = len(texts)
            
            self.optimizer.zero_grad()

            outputs, x, _, lambda_opt = self(**encoded_input.to(self.config.device), encoded_input=encoded_input, idx=idx)
            lambda_opts.extend(lambda_opt.cpu().tolist())
            
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)

            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.config.device)
            loss = self.loss_func(outputs[-bsize:, :], labels)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0:
                logging.info(f"Step {step}, Loss: {total_loss / step*self.config.batch_size}")
                
            labels = None
            outputs = None
            gc.collect()
            torch.cuda.empty_cache()

        # model_name = f"artifacts/models/model_{epoch}.pt"
        # torch.save(
        #     self.state_dict(),
        #     model_name,
        # )
        loss = total_loss / len(dataloader)
        
        return {"loss" : loss}
    
    def run_predict(self, df, df_train, compute_loss=True, embeddings_train=None):
        self = self.to(self.config.device)
        self.eval()
        dataset_test = TextDataset(df.text.values, df.label.values)
        dataloader_test = DataLoader(dataset_test, batch_size=self.config.batch_size_eval, shuffle=False)
        
        train_feats = self.get_train_embeddings(self.df)
        test_feats = self.get_train_embeddings(df)
        
        feats = np.concatenate((train_feats, test_feats))
        labels_ = np.concatenate((self.df.label.values,
                                  [-1]*len(df)))            
        self.fg.build_graph_data(feats, labels_)
        
        preds = []
        feats_test = []
        losses = 0
        lambda_opts = []
        hetef = []
        for idx, texts, labels in tqdm(dataloader_test):
            with torch.no_grad():
                # Tokenize sentences
                encoded_input = self.tokenizer(
                    list(texts), 
                return_tensors='pt',          # Retorna tensores PyTorch
                padding='max_length',         # Preenche até o comprimento máximo
                truncation=True,              # Trunca se o texto for muito longo
                max_length=self.max_length,               # Define o comprimento máximo de tokens
                return_attention_mask=True,   # Retorna a máscara de atenção
                return_token_type_ids=False,  # Não retorna IDs de tipo de token
                add_special_tokens=True       # Adiciona tokens especiais [CLS] e [SEP]
                )
                bsize = len(texts)

                outputs, embeddings_bert, _, lambda_opt = self(**encoded_input.to(self.config.device), encoded_input=encoded_input, idx=idx)
                feats_test.append(embeddings_bert.cpu().numpy())
                lambda_opts.extend(lambda_opt.cpu().tolist())

                predicted_probability, predicted_class = torch.max(
                    outputs[-bsize:, :], dim=1
                )
                preds.extend(predicted_class.cpu().numpy())
                y_pred = self.df.label.tolist() + predicted_class.cpu().tolist()
                # hetef.append(calculate_heterophily_edges(self.fg.edge_index, y_pred))
                if compute_loss:
                    labels = labels.type(torch.LongTensor)
                    loss = self.loss_func(outputs[-bsize:, :], labels.to(self.config.device))
                    losses += loss
        if compute_loss:
            losses = losses/len(dataloader_test)
        else:
            losses = -1
        # logging.info(f"Mean heterophily edge: {sum(hetef)/len(hetef)}")
        # self.plot_lambda(lambda_opts)
        
        return preds, feats, np.concatenate(feats_test), None, losses
  
    def get_train_embeddings(self, df):
        dataset_no_shuffle = TextDataset(df.text, df.label)
        dataloader_no_shuffle = DataLoader(
            dataset_no_shuffle, batch_size=self.config.batch_size, shuffle=False
        )
        feats = []
        self.eval()
        
        for idx, texts, labels in tqdm(dataloader_no_shuffle):
            with torch.no_grad():
                encoded_input = self.tokenizer(
                    list(texts), 
                return_tensors='pt',          # Retorna tensores PyTorch
                padding='max_length',         # Preenche até o comprimento máximo
                truncation=True,              # Trunca se o texto for muito longo
                max_length=self.max_length,               # Define o comprimento máximo de tokens
                return_attention_mask=True,   # Retorna a máscara de atenção
                return_token_type_ids=False,  # Não retorna IDs de tipo de token
                add_special_tokens=True       # Adiciona tokens especiais [CLS] e [SEP]
                )
                output = self.auto_model(**encoded_input.to(self.config.device))  # .pooler_output
                norm_emb = normalize(output, encoded_input, type_emb=self.emb_gnn)
                feats.append(norm_emb.cpu().numpy())
        feats_local = np.concatenate(feats, axis=0)
        return feats_local

def run_experiment2(exp_number, func, settings={}):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perc = 0.01
    
    if "setup" in settings:
        _, _, _, dataset_name = func()
        setup = choose_setup(settings["setup"], dataset_name, perc=perc)
        settings.update(setup)
    
    df_full, df_train, df_compl, df_val, df_test, \
        train_size, target_names, dataset_name = detailed_split(func=func, 
                                                        exp_number=exp_number,
                                                        train_size=settings["sampling_train"],
                                                        # sample_per_class=8,
                                                        test_size=settings["test_size"],
                                                        val_size=settings["val_size"],
                                                        compl_size=settings["compl_size"],
                                                        join_train_test=settings["join_train_test"],
                                                        train_size_per_class=settings["train_size_per_class"],
                                                        )
    num_gen = 2
    augs = ["eda", "keyaug", #"topics"
            ]
    
    # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    # base_name = "stsb-bert-base"
    # base_name = "all-mpnet-base-v1"
    # base_name = "bert-base-nli-mean-tokens"
    # base_name = "msmarco-bert-base-dot-v5"
    base_name = "bert-base-uncased"
    # transformer_name = f"sentence-transformers/{base_name}"
    transformer_name = base_name
    
    df_aug = generate_aug(df_train, num_gen=num_gen, model_name=base_name, augs=augs)
    df_aug["text"] = df_aug["text_aug"]
    df_compl = pd.concat([df_aug, df_compl]).reset_index(drop=True)
    
    check_log_path(dataset_name)
    
    if "dir_model" in settings:
        transformer_name = settings["dir_model"]
        
    dic_max_len = best_max_lenght()
    k = 3
    
    config = {
        ## graph
        "k": k,
        "similar_min": 0.5,
        "mode": "distance",
        "edge_weight": 1,
        "restrictive_class": False,
        ## Text-Bert
        "max_length": dic_max_len[dataset_name],
        ## NN
        "kind_lambda": "concat",
        "num_epochs": 20,
        "batch_size": 16,
        "batch_size_eval": 16,
        "hidden_dim": 256, 
        "dropout_": 0.6,
        "lr_distilbert": 1e-5,
        "lr_gnn": 1e-3,
        "lr_lambda": 0.01,
        "emb_gnn": "avg_norm", # "avg", "avg_norm", "cls_norm", "cls"
        "emb_bert": "cls_norm", # "avg", "avg_norm", "cls_norm", "cls"
        # "input_dim": 768, #384, #768,
        "layer_norm": 1,
        "n_layers": 2,
        # "heads": [2, 1],
        ## loss
        # "optimizer": "AdamWScheduleFree", # AdamWScheduleFree, normal
        "optimizer": "normal", # AdamWScheduleFree, normal
        "m": 0.7,        
        "use_full_train": 0,
        "full_train": 0,
        "dataset_name": dataset_name,
        "exp_number": exp_number,
        "train_size": len(df_train),
        "train_size_p": train_size,
        "test_size": len(df_test),
        "val_size": len(df_val),
        "unlabeled_data": len(df_compl),
        "perc_train": perc,
        "n_classes": len(set(target_names)),
        "device": device,
        "transformer_name": transformer_name,
        "name_model": base_name,
        "pre_train": settings["pre_train"] if "pre_train" in settings else False,
        "num_gen": num_gen,
        "augs": augs,
        "freeze_bert": False
    }
    config["version"] = f"semisup_transd_dist_28-11-24b16_{config['num_epochs']}_ep_{perc}_{base_name}_{config['emb_gnn']}_{config['emb_bert']}"
    config["experiment"] = f"testing_{(config['name_model']).lower()}"
    if "proj" in settings:
        config["proj"] = settings["proj"]
    
    df_full = df_full.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_compl = df_compl.reset_index(drop=True)
    
    # Exemplo de uso
    model = ModelGCN(1)
    model.df_compl = df_compl
    model.report_score = True
    model.setting(config)
    model.report_score = True

    return model.run_experiment(df_train, df_val, df_test, target_names)

def main(config={}):
    funcs_datasets = [
        # get_cstr,
        # get_toy_data,
        # get_trec,
        # get_dmozscience,
        # get_mpqa,
        # get_agnew,
        # get_ohsumed,
        # get_r8, 
        # get_trec_6,
        # get_dblp,
        get_snippets,
        # get_tag_my_news,
        # get_ohsumed_root, 
        # get_ohsumed_title,
        # get_twitter_10k,
        # get_mr,
        # get_r8_double,
        # get_r52,
        # get_20newsgroups,
        # get_dmozscience,get_dmozhealth,get_dmozcomputers,get_dmozsports, 
        # get_syskillwebert, 
        # get_classic4,
        # load_dataset("lex_glue", "scotus"),        
    ]
    funcs_datasets = config["func"] if "func" in config else funcs_datasets
    config["setup"] = "C"
    
    global dataset_name
    global name_method
    dataset_name = ""

    random_states = [
        # 42, 10, 0, 50, 20, 
        11, 35, 8, 3, 23
    ]
    name_method = "semisup_model_gcn_simple"

    return call_exp(funcs_datasets, random_states, 
                    run_experiment2,
                    name_method, settings=config,
                    notify=True)
