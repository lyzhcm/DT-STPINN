"""Training loop for DT-STPINN.

Handles train/validation loops, checkpointing, logging to TensorBoard,
learning rate scheduling, early stopping, and AMP mixed precision.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .loss import DTSTPINNLoss
from .utils.metrics import compute_metrics

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)


class Trainer:
    def __init__(self, model: nn.Module, config, material_props,
                 device: torch.device | None = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = DTSTPINNLoss(config, material_props)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        total_epochs = config.training.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6,
        )

        self.log_dir = Path(config.logging.log_dir) / config.logging.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.save_every = config.logging.save_every
        self.eval_every = config.logging.eval_every
        self.grad_clip = config.training.grad_clip
        self.accumulate_grad = config.training.accumulate_grad_batches
        self.early_stopping_patience = config.training.early_stopping_patience
        self.use_amp = getattr(config.training, "use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.global_step = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int,
                    total_epochs: int | None = None) -> dict:
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        self.optimizer.zero_grad()

        desc = f"Train epoch {epoch}"
        if total_epochs is not None:
            desc = f"Train epoch {epoch}/{total_epochs}"

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in progress:
            batch = self._to_device(batch)

            graph_seq = batch["graph_sequence"]
            if isinstance(graph_seq, list) and len(graph_seq) > 0 and isinstance(graph_seq[0], list):
                graph_seq = graph_seq[0]
            target = batch["target"]
            prev_temp = batch.get("prev_temp", None)
            coords = batch.get("coords", graph_seq[-1].coords)
            edge_index = batch.get("edge_index", graph_seq[-1].edge_index)
            dt = batch.get("dt", 1.0)
            mask = batch.get("target_mask", None)

            if isinstance(dt, torch.Tensor):
                dt = dt.item()

            boundary = getattr(graph_seq[-1], "boundary",
                               torch.zeros(coords.shape[0], device=self.device))
            laser_pos = getattr(graph_seq[-1], "laser_pos",
                                torch.zeros(3, device=self.device))
            is_initial = batch.get("is_initial",
                                    torch.zeros(coords.shape[0], dtype=torch.bool,
                                                device=self.device))

            output = self.model(graph_seq, dt=dt)
            T_pred = output["T_pred"]

            target_time_step = batch.get("target_step",
                                         len(graph_seq) + 1)
            is_initial_step = (target_time_step == 0)
            if is_initial_step:
                is_initial = mask.bool() if mask is not None else None

            with autocast("cuda", enabled=self.use_amp):
                total, components = self.loss_fn.forward(
                    pred=T_pred, target=target, prev_temp=prev_temp,
                    coords=coords, edge_index=edge_index, boundary=boundary,
                    dt=dt, laser_pos=laser_pos, mask=mask,
                    is_initial=is_initial,
                )

            loss = total / self.accumulate_grad
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulate_grad == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            total_loss += total.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

            running_loss = total_loss / (batch_idx + 1)
            postfix = {
                "loss": f"{total.item():.3e}",
                "avg": f"{running_loss:.3e}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            }
            if "T" in components:
                postfix["T"] = f"{components['T'].item():.3e}"
            if "PDE" in components:
                postfix["PDE"] = f"{components['PDE'].item():.3e}"
            progress.set_postfix(postfix)

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches

        return {"loss": avg_loss, **loss_components}

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader,
                       epoch: int | None = None) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        desc = "Validate" if epoch is None else f"Validate epoch {epoch}"
        progress = tqdm(
            val_loader,
            total=len(val_loader),
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            batch = self._to_device(batch)
            graph_seq = batch["graph_sequence"]
            if isinstance(graph_seq, list) and len(graph_seq) > 0 and isinstance(graph_seq[0], list):
                graph_seq = graph_seq[0]
            target = batch["target"]
            prev_temp = batch.get("prev_temp", None)
            coords = batch.get("coords", graph_seq[-1].coords)
            edge_index = batch.get("edge_index", graph_seq[-1].edge_index)
            dt = batch.get("dt", 1.0)
            mask = batch.get("target_mask", None)

            if isinstance(dt, torch.Tensor):
                dt = dt.item()

            boundary = getattr(graph_seq[-1], "boundary",
                               torch.zeros(coords.shape[0], device=self.device))
            laser_pos = getattr(graph_seq[-1], "laser_pos",
                                torch.zeros(3, device=self.device))

            output = self.model(graph_seq, dt=dt)
            T_pred = output["T_pred"]

            total, _ = self.loss_fn.forward(
                pred=T_pred, target=target, prev_temp=prev_temp,
                coords=coords, edge_index=edge_index, boundary=boundary,
                dt=dt, laser_pos=laser_pos, mask=mask,
            )
            total_loss += total.item()
            progress.set_postfix({
                "loss": f"{total.item():.3e}",
                "avg": f"{total_loss / (batch_idx + 1):.3e}",
            })

            all_preds.append(T_pred.detach().cpu())
            all_targets.append(target.detach().cpu())

        avg_loss = total_loss / len(val_loader)
        preds = torch.cat([p.view(-1) for p in all_preds])
        targs = torch.cat([t.view(-1) for t in all_targets])

        valid_mask = ~torch.isnan(targs) & (targs.abs() > 0)
        if valid_mask.any():
            metrics = compute_metrics(preds[valid_mask], targs[valid_mask])
        else:
            metrics = {}

        return {"val_loss": avg_loss, "metrics": metrics}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int | None = None):
        epochs = epochs or self.config.training.epochs

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch, total_epochs=epochs)
            train_time = time.time() - t0

            self._log_metrics(train_metrics, epoch, prefix="train")
            self.writer.add_scalar("time/train_epoch", train_time, epoch)

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", lr, epoch)

            if epoch % self.eval_every == 0:
                val_metrics = self.validate_epoch(val_loader, epoch=epoch)
                self._log_metrics(val_metrics, epoch, prefix="val")

                val_loss = val_metrics["val_loss"]

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    self.epochs_no_improve += 1

                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train: {train_metrics['loss']:.4e} | "
                    f"Val: {val_loss:.4e} | "
                    f"Best: {self.best_val_loss:.4e} @ epoch {self.best_epoch} | "
                    f"Time: {train_time:.1f}s"
                )

                if (self.early_stopping_patience > 0
                        and self.epochs_no_improve >= self.early_stopping_patience):
                    print(f"Early stopping after {epoch} epochs.")
                    break
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train: {train_metrics['loss']:.4e} | "
                    f"Time: {train_time:.1f}s"
                )

            if epoch % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            self.scheduler.step()

        self.writer.close()
        print(f"Training complete. Best val_loss={self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def _log_metrics(self, metrics: dict, epoch: int, prefix: str):
        for key, value in metrics.items():
            if key == "metrics":
                for mk, mv in value.items():
                    self.writer.add_scalar(f"{prefix}/{mk}", mv, epoch)
            else:
                self.writer.add_scalar(f"{prefix}/{key}", value, epoch)

    def save_checkpoint(self, filename: str):
        path = self.log_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_epoch = ckpt["best_epoch"]

    def _to_device(self, batch: dict) -> dict:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, list):
                result[k] = [self._to_device_data(d) for d in v]
            elif hasattr(v, "to"):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _to_device_data(self, data):
        data = data.clone()
        tensor_attrs = [
            "x",
            "y",
            "edge_index",
            "edge_attr",
            "mask",
            "coords",
            "boundary",
            "laser_pos",
        ]
        for attr in tensor_attrs:
            val = getattr(data, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(data, attr, val.to(self.device))
        return data
