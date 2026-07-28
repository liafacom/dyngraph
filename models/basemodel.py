import time 
import os
import datetime
import logging
import pandas as pd
import gc
from models.common import get_configs
from utils.utils import send_msg
import wandb
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
                
    def update(self, dic: dict):
        for key, value in dic.items():
            if isinstance(value, dict):
                value = AttrDict(value)
            self[key] = value
                

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

def trocar_arquivo_log(novo_arquivo):
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    novo_handler = logging.FileHandler(novo_arquivo, mode='a')
    novo_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(novo_handler)
    
class BaseModel:
    def __init__(self, config: dict):
        self.config = AttrDict(config)
        self.results = AttrDict() 
        self.test_every_epoch = True
        self.report_score = False
        
    def setup_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def setup(self):
        self.results = AttrDict() 
        self.machine_config = get_configs()
        self.config["machine"] = self.machine_config["machine"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.save_best_model = False
        try:
            wandb.finish()
            logging.info("Finished previous wandb run")
        except Exception:
            logging.info("Try to finish previous wandb run")
        if "proj" in self.config:
            project = self.config["proj"]
        else:
            project = "TextClassificationDoc"
        self.run = wandb.init(project=project, config=self.config)
    
    def get_metrics(self, y_true, y_pred, tag=""):
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        
        if self.report_score:
            logging.info(classification_report(y_true, y_pred, labels=list(set(self.df_train.label)), 
                                               target_names=self.target_names, zero_division=0))
        
        if tag != "":
            tag = "_"+tag
        
        return {f"acc{tag}": acc, 
                f"f1_micro{tag}": f1_micro, 
                f"f1_macro{tag}": f1_macro}
    
    def logging_setup(self):   
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        t_name = self.config.transformer_name.replace('/','-')
        self.log_filename = f"artifacts/logs/{self.config.dataset_name}/{self.config.class_name}_{t_name}_log_{current_time}.log"
        trocar_arquivo_log(self.log_filename)
        
        for k, v in self.config.items():
            logging.info(f"{k}: {v}")
        
        logging.info(f"Train distribution:")
        logging.info(f"{self.df_train.label.value_counts()}")
        logging.info(f"Val distribution:")
        logging.info(f"{self.df_val.label.value_counts()}")
        logging.info(f"Test distribution:")
        logging.info(f"{self.df_test.label.value_counts()}")
        
            
    def setup_data(self, df_train, df_val, df_test, target_names):
        self.df_train, self.df_val, self.df_test = df_train, df_val, df_test
        self.target_names = target_names
        
    def get_basic_setting(self):
        dir = self.run.dir
        parent_dir = os.path.dirname(dir)
        return (
            f"{self.config}\n"
            f"wandb: \nwandb sync {parent_dir}\n"
                )
    
    def get_name(self):
        if hasattr(self, 'name_model'):
            return f"{self.__class__.__name__}_{self.name_model}"
        return self.__class__.__name__

    def run_experiment(self, df_train, df_val, df_test, target_names):
        gc.collect()
        torch.cuda.empty_cache()
        self.config.class_name = self.get_name()
        self.same_vector = False
        
        self.setup_data(df_train, df_val, df_test, target_names)
        
        self.setup()
        self.logging_setup()     
        self.setup_model()   
        
        content = self.get_basic_setting()
        send_msg(content, user=self.machine_config["machine"], url=self.machine_config["url"])
        
        best_logs = AttrDict()
        best_logs.best_epoch = -1
        best_logs.best_acc = -1
        best_logs.best_f1 = -1
        best_logs.best_val = +float('inf')
        best_logs.best_model = ""
        self.epoch = 0
        
        
        init = datetime.datetime.now()
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logs = AttrDict()
            self.mode_eval = "train"
            log_train = self.run_epoch(epoch)
            logs.update({"loss": log_train["loss"]})
            logging.info(f"Epoch {epoch}, Loss: {log_train['loss']}")
            
            self.mode_eval = "val"
            if "feats" not in log_train:
                log_train["feats"] = None
                
            log_val, embeddings_train, embeddings_val, graph = self.run_val(log_train["feats"])
            logs.update(log_val)
            logging.info(f"Epoch {epoch}, acc_val: {log_val['acc_val']}, val_loss: {log_val['val_loss']}, f1_micro: {log_val['f1_micro_val']}")
            
            log_test = False
            embeddings_test = None
            if self.test_every_epoch:
                self.mode_eval = "test"
                log_test, embeddings_train, embeddings_test, graph = self.run_test(embeddings_train)
                logs.update(log_test)
                logging.info(f"Epoch {epoch}, acc_test: {log_test['acc']}, f1_micro: {log_test['f1_micro_test']}")
             
            if best_logs.best_f1 < logs.f1_micro_val:
                self.set_best(best_logs, logs, epoch, log_test,
                              embeddings_train, embeddings_test)
            elif best_logs.best_f1 == logs.f1_micro_val and best_logs.best_val >= logs.val_loss:
                self.set_best(best_logs, logs, epoch, log_test,
                              embeddings_train, embeddings_test)
                
            self.log_epoch(epoch, logs)
            
        current_day_hour = datetime.datetime.now()
        total = current_day_hour - init
        self.results.update(best_logs)
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.test_every_epoch:
            self.mode_eval = "test"
            log_test, embeddings_train, embeddings_test, graph = self.run_test(embeddings_train)
            logs.update(log_test)
            logging.info(f"Epoch {epoch}, acc_test: {log_test['acc']}, f1_micro: {log_test['f1_micro_test']}")
        
            logging.info("Test metrics")
            logging.info(
                f"Dataset: {self.config.dataset_name}, Best Val - acc: {best_logs.best_acc}, loss: {best_logs.best_val}, f1_micro: {best_logs.best_f1}"
            )
            logging.info(
                f"Total time: {total}, Test metric:"
            )
            for k, v in best_logs.best_test.items():
                if 'test' in k :        
                    logging.info(f"{k}: {v}")
            wandb.config.update(best_logs.best_test)
                
        logging.info(
            f"bestmodel: {best_logs.best_model}, best epoch: {best_logs.best_epoch}"
        )
        dir = self.run.dir
        parent_dir = os.path.dirname(dir)
        logging.info(
            f"wandb sync {parent_dir}"
        )
        total_time = time.time() - start_time
        wandb.config.run_time = total_time
        res = {}
        res.update(wandb.config)
        res.update({
            "best_val_loss": best_logs.best_val,
            "best_val_f1": best_logs.best_f1,
            "best_model": best_logs.best_model,
            "best_epoch": best_logs.best_epoch,
            "wandb_id": self.run.id,
            "wandb_sync": f"wandb sync {parent_dir}"
            })
        wandb.save(self.log_filename)
        if self.save_best_model:
            wandb.save(best_logs.best_model)
        wandb.finish()
        
        return res
    
    def set_best(self, best_logs, logs, epoch, log_test,
                 embeddings_train, embeddings_test):
        model_name = f"artifacts/models/model_{self.config.dataset_name}_{self.config.exp_number}.pt"
        best_logs.best_f1 = logs.f1_micro_val
        best_logs.best_val = logs.val_loss
        best_logs.best_acc = logs.acc_val
        best_logs.best_epoch = epoch
        best_logs.best_model = model_name
        if log_test:
            best_logs.best_test = log_test
            wandb.run.summary["best_accuracy"] = log_test['acc']
            wandb.run.summary["best_f1_micro"] = log_test['f1_micro_test']
        wandb.run.summary["best_epoch"] = epoch
        print_best_logs = best_logs.copy()
        if log_test:
            del print_best_logs["best_test"]["prediction"]
        logging.info(f"Current best: {print_best_logs}")
        
    def run_epoch(self, epoch):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def run_predict(self, df, df_train, compute_loss=True, embeddings_train=None):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def run_val(self, embedding_train=None):
        y_pred, embedding_train, embedding_val, graph, loss = self.run_predict(self.df_val, self.df_train, True, embedding_train)
        dic_res = {}
        dic_res['predict_val'] = y_pred
        dic_res['val_loss'] = loss
        dic_res.update(self.get_metrics(self.df_val.label.tolist(), y_pred, tag="val"))
        return dic_res, embedding_train, embedding_val, graph
    
    def run_test(self, embeddings_train=None):
        if self.config.use_full_train:
            df = pd.concat([self.df_train, self.df_val]).reset_index(drop=True)
            y_pred, embedding_train, embedding_test, graph, _ = self.run_predict(self.df_test, df, False, embeddings_train)
        else:
            y_pred, embedding_train, embedding_test, graph, _ = self.run_predict(self.df_test, self.df_train, False, embeddings_train)
            
        dic_res = {}
        dic_res['prediction'] = y_pred
        dic_res.update(self.get_metrics(self.df_test.label.tolist(), y_pred, tag="test"))
        dic_res.update(self.get_metrics(self.df_test.label.tolist(), y_pred, tag=""))
        
        return dic_res, embedding_train, embedding_test, graph
    
    def log_epoch(self, epoch, metrics, additional_metrics=None):
        metrics.epoch = epoch
        if additional_metrics:
            metrics.update(additional_metrics)
        wandb.log(metrics)
