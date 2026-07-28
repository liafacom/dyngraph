FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace/dyngraphbert
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV WANDB_MODE=disabled
CMD ["python", "run_experiment.py", "--help"]
