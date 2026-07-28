# DynGraphBERT

Repositório reprodutível dos experimentos semi-supervisionados com DynGraphBERT. O método principal está implementado em `models/imodel_simple_gcn_semisup.py`.

O trabalho foi publicado na revista *Informatics*: [DynGraph-BERT: Combining BERT and GNN Using Dynamic Graphs for Inductive Semi-Supervised Text Classification](https://www.mdpi.com/2227-9709/12/1/20).

## Datasets

O escopo está limitado a Ohsumed, R8, AG News, Web Snippets e DBLP. Ohsumed, R8, Web Snippets e DBLP estão armazenados em `datasets/`. O AG News é baixado automaticamente pelo `torchtext` na primeira execução.

## Execução com Docker

```bash
docker build -t dyngraphbert .
docker run --rm --gpus all \
  -v "$PWD/artifacts:/workspace/dyngraphbert/artifacts" \
  dyngraphbert \
  python run_experiment.py --dataset ohsumed --setup C --seeds 11 35 8 3 23
```

Para executar todos os datasets e as sementes usadas na pesquisa:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifacts:/workspace/dyngraphbert/artifacts" \
  dyngraphbert python run_experiment.py
```

## Execução local

Requer Python 3.10 e uma instalação CUDA compatível:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --dataset r8
```

Consulte todos os argumentos com `python run_experiment.py --help`.

## Weights & Biases

O W&B fica desabilitado por padrão. Para habilitá-lo:

```bash
wandb login
python run_experiment.py --dataset agnews --wandb --wandb-project DynGraphBERT
```

O arquivo `config.json` é opcional. Para configurar uma notificação por webhook (Discord), copie `config.example.json` para `config.json` e preencha `url`.

## Resultados

Cada execução grava versões CSV, JSON e pickle em `artifacts/results/`. Logs, modelos e embeddings também são mantidos sob `artifacts/` e não são versionados.

## Citação

Se este trabalho for útil em sua pesquisa, por favor cite o nosso artigo:

```bibtex
@Article{informatics12010020,
  AUTHOR = {Perin, Eliton Luiz Scardin and Souza, Mariana Caravanti de and Silva, Jonathan de Andrade and Matsubara, Edson Takashi},
  TITLE = {DynGraph-BERT: Combining BERT and GNN Using Dynamic Graphs for Inductive Semi-Supervised Text Classification},
  JOURNAL = {Informatics},
  VOLUME = {12},
  YEAR = {2025},
  NUMBER = {1},
  ARTICLE-NUMBER = {20},
  URL = {https://www.mdpi.com/2227-9709/12/1/20},
  ISSN = {2227-9709},
  ABSTRACT = {The combination of Bidirecional Encoder Representations from Transformers (BERT) and Graph Neural Networks (GNNs) has been extensively explored in the text classification literature, usually employing BERT as a feature extractor combined with heterogeneous static graphs. BERT transfers information via token embeddings, which are propagated through GNNs. Text-specific information defines a static heterogeneous graph. Static graphs represent specific relationships and do not have the flexibility to add new knowledge to the graph. To address this issue, we build a tied connection between BERT and GNN exclusively using token embeddings to define the graph and propagate the embeddings, which can force the BERT to redefine the GNN graph topology to improve accuracy. Thus, in this study, we re-examine the design spaces and test the limits of what this pure homogeneous graph using BERT embeddings can achieve. Homogeneous graphs offer structural simplicity and greater generalization capabilities, particularly when integrated with robust representations like those provided by BERT. To improve accuracy, the proposed approach also incorporates text augmentation and label propagation at test time. Experimental results show that the proposed method outperforms state-of-the-art methods across all datasets analyzed, with consistent accuracy improvements as more labeled examples are included.},
  DOI = {10.3390/informatics12010020}
}
```
