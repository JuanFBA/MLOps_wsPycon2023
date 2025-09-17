# src/data/load.py
import argparse
import os
import json

import torch
from torch.utils.data import TensorDataset

import wandb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
parser.add_argument('--train-size', type=float, default=0.8, help='Fracción para train vs test')
parser.add_argument('--val-ratio', type=float, default=0.2, help='Fracción del train para validación')
parser.add_argument('--seed', type=int, default=42, help='Semilla de aleatoriedad')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")


def load(train_size: float = 0.8, val_ratio: float = 0.2, seed: int = 42):
    """
    Carga California Housing, hace split train/val/test y devuelve TensorDatasets.
    - target: 'MedHouseVal'
    - X (features) y y (target) quedan en float32 (regresión).
    """
    # 1) Cargar dataset en un DataFrame
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # incluye la columna objetivo 'MedHouseVal'

    target_col = data.target_names[0]  # normalmente 'MedHouseVal'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2) Split train/test y luego train/val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=seed
    )

    # 3) Convertir a tensores float32 (regresión)
    def to_tensor(a):
        return torch.tensor(a.to_numpy(dtype=np.float32))

    x_train_t = to_tensor(X_train)
    x_val_t   = to_tensor(X_val)
    x_test_t  = to_tensor(X_test)

    # y como vector columna float32
    y_train_t = torch.tensor(y_train.to_numpy(dtype=np.float32)).view(-1, 1)
    y_val_t   = torch.tensor(y_val.to_numpy(dtype=np.float32)).view(-1, 1)
    y_test_t  = torch.tensor(y_test.to_numpy(dtype=np.float32)).view(-1, 1)

    training_set   = TensorDataset(x_train_t, y_train_t)
    validation_set = TensorDataset(x_val_t,  y_val_t)
    test_set       = TensorDataset(x_test_t,  y_test_t)

    meta = {
        "source": "sklearn.datasets.fetch_california_housing",
        "target": target_col,
        "n_features": X.shape[1],
        "features": list(X.columns),
        "sizes": [len(training_set), len(validation_set), len(test_set)],
        "splits": {"train_size": train_size, "val_ratio": val_ratio, "seed": seed},
    }
    return [training_set, validation_set, test_set], meta


def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps-Pycon2023"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"Load Raw Data ExecId-{args.IdExecution}",
        job_type="load-data",
    ) as run:

        datasets, meta = load(train_size=args.train_size, val_ratio=args.val_ratio, seed=args.seed)
        names = ["training", "validation", "test"]

        # 🏺 crear Artifact con metadatos del dataset tabular
        raw_data = wandb.Artifact(
            "california-housing-raw",
            type="dataset",
            description="California Housing dataset split into train/val/test",
            metadata=meta,
        )

        # Guardar tensores como archivos .pt dentro del artifact
        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # (Opcional) guardar features.json dentro del artifact
        with raw_data.new_file("features.json", mode="w") as f:
            json.dump(meta, f, indent=2)

        # ✍️ Subir el artifact a W&B
        run.log_artifact(raw_data)


# testing
if __name__ == "__main__":
    load_and_log()
