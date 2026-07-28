import pandas as pd
import gc
import logging
from models.basemodel import AttrDict, BaseModel
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from BertGNN.data2graph import FAISSGraphBuilder
from models.common import add_gaussian_noise, detailed_split, generate_aug, TextDataset
from utils.utils import best_max_length, check_log_path, choose_setup


def mean_pooling(model_output, attention_mask):
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
        ]
        if "norm" in type_emb:
            embeddings_bert = F.normalize(embeddings_bert, p=2, dim=1)
        return embeddings_bert
    if attention_mask is None and encoded_input is not None:
        attention_mask = encoded_input["attention_mask"]

    sentence_embeddings = mean_pooling(model_output, attention_mask)
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
                                       ))        
            for n in range(n_layers-1):
                self.gnns.append(GATv2Conv(heads[n]*hidden_dim, hidden_dim, heads=heads[n+1], bias=True, 
                                           ))
            self.gnns.append(GATv2Conv(heads[-2] * hidden_dim, self.numero_de_classes, 
                                    heads=heads[-1], bias=False,
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
    
    def forward(self, input_ids, encoded_input, idx, token_type_ids=None, attention_mask=None):
        outputs = self.auto_model(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        
        embeddings_gnn = normalize(outputs, encoded_input, type_emb=self.emb_gnn)
        embeddings_bert = normalize(outputs, encoded_input, type_emb=self.emb_bert)

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
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.loss_func = F.nll_loss
        
    def run_epoch(self, epoch):
        total_loss = 0
        step = 0
        
        self.df = pd.concat([self.df_train, self.df_compl, self.df_test])
        self.df = self.df.reset_index(drop=True)
        feats_local = self.get_train_embeddings(self.df)
        gc.collect()
        torch.cuda.empty_cache()
        self.fg.build_graph_data(feats_local, self.df.label.values)
        df_training  = pd.concat([self.df_train, self.df_compl])
        df_training = df_training.reset_index(drop=True)
        dataset = TextDataset(df_training.text, df_training.label)
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        lambda_opts = []
        self.train()
        for idx, texts, labels in tqdm(dataloader):
            step += 1
            encoded_input = self.tokenizer(
                list(texts), 
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=True
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
        for idx, texts, labels in tqdm(dataloader_test):
            with torch.no_grad():
                encoded_input = self.tokenizer(
                    list(texts), 
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=True
                )
                bsize = len(texts)

                outputs, embeddings_bert, _, lambda_opt = self(**encoded_input.to(self.config.device), encoded_input=encoded_input, idx=idx)
                feats_test.append(embeddings_bert.cpu().numpy())
                lambda_opts.extend(lambda_opt.cpu().tolist())

                predicted_probability, predicted_class = torch.max(
                    outputs[-bsize:, :], dim=1
                )
                preds.extend(predicted_class.cpu().numpy())
                if compute_loss:
                    labels = labels.type(torch.LongTensor)
                    loss = self.loss_func(outputs[-bsize:, :], labels.to(self.config.device))
                    losses += loss
        if compute_loss:
            losses = losses/len(dataloader_test)
        else:
            losses = -1
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
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=True
                )
                output = self.auto_model(**encoded_input.to(self.config.device))
                norm_emb = normalize(output, encoded_input, type_emb=self.emb_gnn)
                feats.append(norm_emb.cpu().numpy())
        feats_local = np.concatenate(feats, axis=0)
        return feats_local

def run_experiment2(exp_number, func, settings=None):
    settings = {} if settings is None else settings
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
                                                        test_size=settings["test_size"],
                                                        val_size=settings["val_size"],
                                                        compl_size=settings["compl_size"],
                                                        join_train_test=settings["join_train_test"],
                                                        train_size_per_class=settings["train_size_per_class"],
                                                        )
    num_gen = 2
    augs = ["eda", "keyaug"]
    base_name = "bert-base-uncased"
    transformer_name = base_name
    
    df_aug = generate_aug(df_train, num_gen=num_gen)
    df_aug["text"] = df_aug["text_aug"]
    df_compl = pd.concat([df_aug, df_compl]).reset_index(drop=True)
    
    check_log_path(dataset_name)
    
    if "dir_model" in settings:
        transformer_name = settings["dir_model"]
        
    max_lengths = best_max_length()
    k = 3
    
    config = {
        "k": k,
        "similar_min": 0.5,
        "mode": "distance",
        "edge_weight": 1,
        "restrictive_class": False,
        "max_length": max_lengths[dataset_name],
        "kind_lambda": "concat",
        "num_epochs": 20,
        "batch_size": 16,
        "batch_size_eval": 16,
        "hidden_dim": 256, 
        "dropout_": 0.6,
        "lr_distilbert": 1e-5,
        "lr_gnn": 1e-3,
        "lr_lambda": 0.01,
        "emb_gnn": "avg_norm",
        "emb_bert": "cls_norm",
        "layer_norm": 1,
        "n_layers": 2,
        "optimizer": "normal",
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
    
    model = ModelGCN(1)
    model.df_compl = df_compl
    model.report_score = True
    model.setting(config)
    model.report_score = True

    return model.run_experiment(df_train, df_val, df_test, target_names)
