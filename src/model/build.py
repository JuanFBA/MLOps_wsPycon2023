# src/model/build.py
import os
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset
import wandb

# ---------------------------
#   Modelo Regressor (MLP)
# ---------------------------
class Regressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int] = [128, 64]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]  # salida escalar (regresión)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
#   Utilidades
# ---------------------------
def _read_split_pt(dirpath: str, split: str) -> TensorDataset:
    x, y = torch.load(os.path.join(dirpath, f"{split}.pt"))
    return TensorDataset(x, y)

# ---------------------------
#   CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="ID of the execution")
parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64],
                    help="Capas ocultas del MLP")
parser.add_argument("--artifact_in", type=str,
                    default="california-housing-preprocess:latest",
                    help="Artifact con los datos preprocesados para inferir in_dim")
parser.add_argument("--model_name", type=str, default="california-regressor-init",
                    help="Nombre del artifact de modelo inicial")
parser.add_argument("--model_desc", type=str, default="Initialized MLP Regressor",
                    help="Descripción del artifact de modelo inicial")
args = parser.parse_args()
if not args.IdExecution:
    args.IdExecution = "testing console"

os.makedirs("./models", exist_ok=True)

# ---------------------------
#   Build + log a W&B
# ---------------------------
def build_model_and_log():
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps-Pycon2023"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"initialize Model ExecId-{args.IdExecution}",
        job_type="initialize-model",
        config={"hidden": args.hidden},
    ) as run:

        # Tomamos el artifact PREPROCESADO para conocer in_dim
        data_art = run.use_artifact(args.artifact_in, type="dataset")
        data_dir = data_art.download()
        train_ds = _read_split_pt(data_dir, "training")
        in_dim = train_ds.tensors[0].shape[1]

        # Construimos el modelo
        model = Regressor(in_dim=in_dim, hidden=list(args.hidden))
        ckpt_path = "./models/initialized_regressor.pth"
        torch.save(
            {"model_state": model.state_dict(), "in_dim": in_dim, "hidden": list(args.hidden)},
            ckpt_path,
        )

        # Publicamos artifact de modelo inicial
        model_artifact = wandb.Artifact(
            args.model_name,
            type="model",
            description=args.model_desc,
            metadata={"in_dim": in_dim, "hidden": list(args.hidden)},
        )
        model_artifact.add_file(ckpt_path)
        run.log_artifact(model_artifact)

if __name__ == "__main__":
    build_model_and_log()
