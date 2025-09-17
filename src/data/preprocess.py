# src/data/preprocess.py
import os
import argparse
import json
import torch
from torch.utils.data import TensorDataset
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
parser.add_argument('--standardize', type=bool, default=True, help='Estandarizar X con media y desv. del train')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"


def read(data_dir, split):
    """Lee un split ('training'|'validation'|'test') guardado como .pt"""
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    # Asegurar tipos float32 (regresión)
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    return TensorDataset(x, y)


def _fit_standardizer(x: torch.Tensor):
    """Calcula media y std por columna (usa solo TRAIN). Evita std=0."""
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.tensor(1e-8, dtype=std.dtype, device=std.device), std)
    return mean, std


def _apply_standardizer(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (x - mean) / std


def preprocess_and_log(standardize: bool):
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps-Pycon2023"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"Preprocess Data ExecId-{args.IdExecution}",
        job_type="preprocess-data"
    ) as run:

        # Artifact de entrada (producido por load.py)
        raw_data_artifact = run.use_artifact('california-housing-raw:latest', type='dataset')
        raw_dataset_dir = raw_data_artifact.download(root="./data/artifacts/")

        # Leer splits crudos
        train_raw = read(raw_dataset_dir, "training")
        val_raw   = read(raw_dataset_dir, "validation")
        test_raw  = read(raw_dataset_dir, "test")

        x_train, y_train = train_raw.tensors
        x_val,   y_val   = val_raw.tensors
        x_test,  y_test  = test_raw.tensors

        meta = {
            "source_artifact": "california-housing-raw:latest",
            "steps": {"standardize": standardize},
        }

        # Ajustar estandarizador en TRAIN y aplicar a todos los splits
        if standardize:
            mean, std = _fit_standardizer(x_train)
            x_train = _apply_standardizer(x_train, mean, std)
            x_val   = _apply_standardizer(x_val, mean, std)
            x_test  = _apply_standardizer(x_test, mean, std)
            meta["standardizer"] = {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}
        else:
            mean = std = None

        # Crear artifact de salida
        processed_data = wandb.Artifact(
            "california-housing-preprocess",
            type="dataset",
            description="California Housing preprocesado (z-score con stats de train)",
            metadata=meta
        )

        # Guardar los splits procesados
        for split_name, (xx, yy) in {
            "training": (x_train, y_train),
            "validation": (x_val, y_val),
            "test": (x_test, y_test),
        }.items():
            with processed_data.new_file(split_name + ".pt", mode="wb") as f:
                torch.save((xx, yy), f)

        # Guardar stats del estandarizador (útil para inferencia)
        if standardize:
            with processed_data.new_file("standardizer.pt", mode="wb") as f:
                torch.save({"mean": mean, "std": std}, f)

        with processed_data.new_file("preprocess_meta.json", mode="w") as f:
            json.dump(meta, f, indent=2)

        run.log_artifact(processed_data)


if __name__ == "__main__":
    preprocess_and_log(standardize=args.standardize)