# src/model/train.py
import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import wandb

# ---------------------------
#   Args / Device
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# ---------------------------
#   Utilidades de datos
# ---------------------------
def read(data_dir: str, split: str) -> TensorDataset:
    """Lee un split ('training'|'validation'|'test') guardado como .pt"""
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    # asegurar float32 (regresión) y 2D
    x = x.to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.view(-1, 1)
    y = y.to(dtype=torch.float32).view(-1, 1)
    return TensorDataset(x, y)

# ---------------------------
#   Modelo: Regressor MLP
# ---------------------------
class Regressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int] = [128, 64]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]  # salida escalar
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, in_dim)
        return self.net(x)

# ---------------------------
#   Entrenamiento / Eval
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    ys = []
    yhs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.append(y.cpu().numpy())
        yhs.append(y_hat.cpu().numpy())
    mean_loss = total_loss / max(n, 1)
    y_true = np.concatenate(ys, axis=0).ravel()
    y_pred = np.concatenate(yhs, axis=0).ravel()
    mae = np.mean(np.abs(y_true * 1.0 - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2 manual (para no depender de sklearn)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mean_loss, mae, rmse, r2

# ---------------------------
#   Pipeline con W&B
# ---------------------------
def train_and_log():
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps-Pycon2023"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"Train Model ExecId-{args.IdExecution}",
        job_type="train-model",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
        },
    ) as run:
        cfg = wandb.config

        # 1) Dataset preprocesado (artifact de preprocess.py)
        data_art = run.use_artifact("california-housing-preprocess:latest", type="dataset")
        data_dir = data_art.download()
        train_ds = read(data_dir, "training")
        val_ds   = read(data_dir, "validation")

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

        # 2) Modelo
        in_dim = train_ds.tensors[0].shape[1]
        model = Regressor(in_dim=in_dim, hidden=list(cfg.hidden)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.MSELoss()

        # 3) Loop de entrenamiento
        best_rmse = float("inf")
        os.makedirs("models", exist_ok=True)
        best_path = "models/regressor_best.pth"

        for epoch in range(cfg.epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, mae, rmse, r2 = evaluate(model, val_loader, criterion)

            wandb.log({
                "epoch": epoch,
                "train/loss_mse": tr_loss,
                "val/loss_mse": val_loss,
                "val/mae": mae,
                "val/rmse": rmse,
                "val/r2": r2,
            })

            if rmse < best_rmse:
                best_rmse = rmse
                torch.save({"model_state": model.state_dict(),
                            "in_dim": in_dim,
                            "hidden": list(cfg.hidden)}, best_path)

        # 4) Subir modelo entrenado como Artifact
        model_art = wandb.Artifact(
            "california-regressor",
            type="model",
            description="Regressor MLP entrenado en California Housing",
            metadata={"in_dim": in_dim, "hidden": list(cfg.hidden)}
        )
        model_art.add_file(best_path)
        run.log_artifact(model_art)

        run.summary.update({"best_val_rmse": best_rmse})

def evaluate_and_log():
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "MLOps-Pycon2023"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"Eval Model ExecId-{args.IdExecution}",
        job_type="eval-model",
    ) as run:
        # Datos
        data_art = run.use_artifact("california-housing-preprocess:latest", type="dataset")
        data_dir = data_art.download()
        test_ds = read(data_dir, "test")
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        # Modelo
        model_art = run.use_artifact("california-regressor:latest", type="model")
        model_dir = model_art.download()
        ckpt = torch.load(os.path.join(model_dir, "regressor_best.pth"), map_location=device)
        model = Regressor(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"]).to(device)
        model.load_state_dict(ckpt["model_state"])

        # Métricas
        criterion = nn.MSELoss()
        test_loss, mae, rmse, r2 = evaluate(model, test_loader, criterion)

        run.summary.update({
            "test/loss_mse": test_loss,
            "test/mae": mae,
            "test/rmse": rmse,
            "test/r2": r2,
        })
        wandb.log({
            "test/loss_mse": test_loss,
            "test/mae": mae,
            "test/rmse": rmse,
            "test/r2": r2,
        })

# ---------------------------
#   Ejecución
# ---------------------------
if __name__ == "__main__":
    train_and_log()
    evaluate_and_log()
