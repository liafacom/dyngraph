# DynGraphBERT

Repositório reprodutível dos experimentos semi-supervisionados com DynGraphBERT. O método principal está implementado em `models/imodel_simple_gcn_semisup.py`.

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

O arquivo `config.json` é opcional. Para configurar uma notificação por webhook, copie `config.example.json` para `config.json` e preencha `url`. Não versione credenciais.

## Resultados

Cada execução grava versões CSV, JSON e pickle em `artifacts/results/`. Logs, modelos e embeddings também são mantidos sob `artifacts/` e não são versionados.

## Proveniência

Os arquivos centrais foram extraídos da versão commitada do repositório de pesquisa, ignorando deliberadamente alterações locais não commitadas. A CLI e os ajustes de configuração foram adicionados para tornar a execução isolada e reproduzível.
