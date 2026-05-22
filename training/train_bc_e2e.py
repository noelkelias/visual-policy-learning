import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# DATASET
# =========================
class PandaBCDataset(Dataset):
    def __init__(self, path):

        with h5py.File(path, "r") as f:
            self.images = f["observations"][:]   # (N, H, W, 3)
            self.actions = f["actions"][:]       # (N, action_dim)

        self.images = self.images.astype(np.float32) / 255.0
        self.actions = self.actions.astype(np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):

        img = torch.from_numpy(
            self.images[idx]
        ).permute(2, 0, 1).contiguous()

        act = torch.from_numpy(
            self.actions[idx]
        ).contiguous()

        return img, act


# =========================
# POLICIES
# =========================
from policies.cnn_policy import CNNPolicy
from policies.clip_policy import CLIPPolicy
from policies.dinov2_policy import DINOv2Policy


def build_policy(name, action_dim, device):

    if name == "cnn":
        return CNNPolicy(action_dim=action_dim).to(device)

    elif name == "clip":
        return CLIPPolicy(action_dim=action_dim).to(device)

    elif name == "dinov2":
        return DINOv2Policy(action_dim=action_dim).to(device)

    else:
        raise ValueError(f"Unknown policy: {name}")


# =========================
# VALIDATION
# =========================
@torch.no_grad()
def evaluate(policy, loader, loss_fn, device):

    policy.eval()

    total_loss = 0.0

    for imgs, actions in loader:

        imgs = imgs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):
            pred_actions = policy(imgs)
            loss = loss_fn(pred_actions, actions)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    policy.train()

    return avg_loss


# =========================
# TRAIN LOOP
# =========================
def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------
    # Dataset
    # ---------------------
    full_dataset = PandaBCDataset(args.data)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    indices = np.arange(len(full_dataset))

    # deterministic split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(
        full_dataset,
        train_indices
    )

    val_dataset = Subset(
        full_dataset,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # ---------------------
    # Model
    # ---------------------
    policy = build_policy(
        args.policy,
        args.action_dim,
        device
    )

    # Freeze CLIP encoder
    if args.policy == "clip":
        for p in policy.encoder.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr
    )

    loss_fn = nn.MSELoss()

    use_amp = (device.type == "cuda")

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    # =====================
    # OUTPUT SETUP
    # =====================

    checkpoint_dir = os.path.join(
        "models",
        "checkpoints",
        args.run_name
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    best_model_path = os.path.join(
        "models",
        f"{args.run_name}.pt"
    )

    log_path = os.path.join(
        checkpoint_dir,
        "logs.csv"
    )

    # CSV header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "gap",
            "best_val",
            "lr"
        ])

    # =====================
    # START INFO
    # =====================

    print(f"\nTraining: {args.run_name}")
    print(f"Policy: {args.policy}")
    print(f"Device: {device}")
    print(f"Total samples: {len(full_dataset)}")
    print(
        f"Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)}"
    )
    print()

    best_val_loss = float("inf")
    best_epoch = -1

    # =====================
    # TRAINING
    # =====================
    for epoch in range(args.epochs):

        policy.train()

        total_train_loss = 0.0

        # -----------------
        # TRAIN
        # -----------------
        for imgs, actions in train_loader:

            imgs = imgs.to(
                device,
                non_blocking=True
            )

            actions = actions.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                pred_actions = policy(imgs)

                loss = loss_fn(
                    pred_actions,
                    actions
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = (
            total_train_loss / len(train_loader)
        )

        # -----------------
        # VALIDATION
        # -----------------
        avg_val_loss = evaluate(
            policy,
            val_loader,
            loss_fn,
            device
        )

        gap = avg_val_loss - avg_train_loss

        # -----------------
        # SAVE BEST MODEL
        # -----------------
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "policy": args.policy,
                },
                best_model_path
            )

        # -----------------
        # SAVE CHECKPOINTS
        # -----------------
        if args.save_every > 0 and (
            epoch % args.save_every == 0
        ):

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"epoch_{epoch:03d}.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "policy": args.policy,
                },
                checkpoint_path
            )

        # -----------------
        # CONSOLE LOGGING
        # -----------------
        print(
            f"[{epoch:03d}] "
            f"train={avg_train_loss:.4f} "
            f"val={avg_val_loss:.4f} "
            f"best={best_val_loss:.4f}"
        )

        # -----------------
        # CSV LOGGING
        # -----------------
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                avg_train_loss,
                avg_val_loss,
                gap,
                best_val_loss,
                optimizer.param_groups[0]["lr"]
            ])

    # =====================
    # FINAL SUMMARY
    # =====================

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val  : {best_val_loss:.6f}")
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
        "--run_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=["cnn", "clip", "dinov2"],
        required=True
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=7
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=5
    )

    args = parser.parse_args()

    train(args)