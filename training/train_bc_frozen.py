# training/train_bc_frozen.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import csv
import h5py
import argparse

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from policies.action_head import ActionHead


# =========================
# DATASET
# =========================
class FeatureDataset(Dataset):

    def __init__(self, path):

        self.h5 = h5py.File(path, "r")

        self.features = self.h5["features"]
        self.actions = self.h5["actions"]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):

        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.actions[idx]),
        )


# =========================
# TRAIN
# =========================
def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = FeatureDataset(args.data)

    n = len(dataset)

    train_size = int(0.8 * n)

    train_set = Subset(
        dataset,
        list(range(train_size))
    )

    val_set = Subset(
        dataset,
        list(range(train_size, n))
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # =========================
    # MODEL
    # =========================

    sample_feat, _ = dataset[0]

    input_dim = sample_feat.shape[0]

    model = ActionHead(
        input_dim=input_dim,
        action_dim=args.action_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    loss_fn = nn.SmoothL1Loss(beta=0.02)

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # =========================
    # PATHS
    # =========================

    checkpoint_dir = os.path.join(
        "models", "checkpoints", args.model_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = f"models/{args.model_name}.pt"

    log_path = os.path.join(checkpoint_dir, "logs.csv")

    # CSV HEADER
    with open(log_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "gap",
            "best_val",
            "lr",
        ])

    print(f"\nTraining: {args.model_name}")
    print(f"Input dim: {input_dim}")

    print(
        f"Train samples: {len(train_set)} "
        f"| Val samples: {len(val_set)}\n"
    )

    best_val = float("inf")
    best_epoch = -1

    # =========================
    # LOOP
    # =========================

    for epoch in range(args.epochs):

        # =====================
        # TRAIN
        # =====================

        model.train()

        train_loss = 0.0

        for feats, actions in train_loader:

            feats = feats.to(device).float()
            actions = actions.to(device).float()

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):

                pred = model(feats)

                loss = loss_fn(
                    pred,
                    actions
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for feats, actions in val_loader:

                feats = feats.to(device).float()

                actions = actions.to(device).float()

                pred = model(feats)

                loss = loss_fn(
                    pred,
                    actions
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        gap = val_loss - train_loss

        scheduler.step()

        # =====================
        # SAVE BEST
        # =====================

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "action_dim": args.action_dim,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_model_path,
            )

        # =====================
        # PERIODIC CHECKPOINTS
        # =====================

        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"epoch_{epoch:03d}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "action_dim": args.action_dim,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        # =====================
        # PRINT
        # =====================

        print(
            f"[{epoch:03d}] "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"best={best_val:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # =====================
        # LOG CSV
        # =====================

        with open(log_path, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                gap,
                best_val,
                optimizer.param_groups[0]["lr"],
            ])

    # =========================
    # FINAL SUMMARY
    # =========================

    print("\nTraining complete.")

    print(f"Best epoch: {best_epoch}")

    print(f"Best val  : {best_val:.6f}")

    print(f"Model     : {best_model_path}")

    print(f"Logs      : {log_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save epoch_XXX.pt every N epochs (0 to disable)",
    )

    args = parser.parse_args()

    train(args)