"""Training loop for DT-STPINN.

Handles train/validation loops, checkpointing, logging to TensorBoard,
learning rate scheduling, early stopping, and AMP mixed precision.
"""
from __future__ import annotations

import gc
import random
import re
import time
from pathlib import Path

import numpy as np
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
        requested_amp_dtype = getattr(config.training, "amp_dtype", "auto").lower()
        if requested_amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError(
                "training.amp_dtype must be auto, float16, or bfloat16."
            )

        self.amp_enabled = self.use_amp and self.device.type == "cuda"
        if requested_amp_dtype == "auto":
            use_bfloat16 = self.amp_enabled and torch.cuda.is_bf16_supported()
            self.amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        else:
            self.amp_dtype = getattr(torch, requested_amp_dtype)

        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=scaler_enabled)
        self.amp_dtype_name = str(self.amp_dtype).removeprefix("torch.")

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.global_step = 0
        self.current_epoch = 0

    def _cuda_synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _optimizer_step(self):
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.global_step += 1

    def train_epoch(self, train_loader: DataLoader, epoch: int,
                    total_epochs: int | None = None) -> dict:
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        batch_times = []
        self.optimizer.zero_grad()

        num_batches = len(train_loader)
        if num_batches == 0:
            raise RuntimeError("Training DataLoader is empty.")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

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
            self._cuda_synchronize()
            batch_start = time.perf_counter()
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

            target_time_step = batch.get("target_step",
                                         len(graph_seq) + 1)
            is_initial_step = (target_time_step == 0)
            if is_initial_step:
                is_initial = mask.bool() if mask is not None else None

            with autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                output = self.model(graph_seq, dt=dt)
                T_pred = output["T_pred"]
                total, components = self.loss_fn.forward(
                    pred=T_pred, target=target, prev_temp=prev_temp,
                    coords=coords, edge_index=edge_index, boundary=boundary,
                    dt=dt, laser_pos=laser_pos, mask=mask,
                    is_initial=is_initial,
                )

            accumulation_start = (batch_idx // self.accumulate_grad) * self.accumulate_grad
            accumulation_size = min(
                self.accumulate_grad, num_batches - accumulation_start
            )
            loss = total / accumulation_size
            self.scaler.scale(loss).backward()

            is_accumulation_end = (
                (batch_idx + 1) % self.accumulate_grad == 0
                or batch_idx + 1 == num_batches
            )
            if is_accumulation_end:
                self._optimizer_step()

            self._cuda_synchronize()
            batch_times.append(time.perf_counter() - batch_start)

            total_loss += total.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

            running_loss = total_loss / (batch_idx + 1)
            postfix = {
                "loss": f"{total.item():.3e}",
                "avg": f"{running_loss:.3e}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "s/b": f"{batch_times[-1]:.1f}",
            }
            if "T" in components:
                postfix["T"] = f"{components['T'].item():.3e}"
            if "PDE" in components:
                postfix["PDE"] = f"{components['PDE'].item():.3e}"
            progress.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches

        peak_vram_gb = 0.0
        if self.device.type == "cuda":
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        return {
            "loss": avg_loss,
            **loss_components,
            "seconds_per_batch": sum(batch_times) / len(batch_times),
            "peak_vram_gb": peak_vram_gb,
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader,
                       epoch: int | None = None) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        worst_case = None

        if len(val_loader) == 0:
            raise RuntimeError("Validation DataLoader is empty.")

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

            with autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
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

            pred_flat = T_pred.detach().float().reshape(-1)
            target_flat = target.detach().float().reshape(-1)
            valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)

            if mask is not None:
                active_mask = mask.detach().bool().reshape(-1)
                if active_mask.numel() != target_flat.numel():
                    if target_flat.numel() % active_mask.numel() != 0:
                        raise ValueError(
                            "target_mask cannot be aligned with validation targets: "
                            f"{active_mask.numel()} mask values for "
                            f"{target_flat.numel()} target values."
                        )
                    repeats = target_flat.numel() // active_mask.numel()
                    active_mask = active_mask.repeat_interleave(repeats)
                valid_mask &= active_mask

            if valid_mask.any():
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
                valid_preds = pred_flat[valid_mask]
                valid_targets = target_flat[valid_mask]
                all_preds.append(valid_preds.cpu())
                all_targets.append(valid_targets.cpu())

                abs_error = (valid_preds - valid_targets).abs()
                local_max_pos = int(abs_error.argmax().item())
                local_max_error = float(abs_error[local_max_pos].item())
                if worst_case is None or local_max_error > worst_case["abs_error"]:
                    flat_index = int(valid_indices[local_max_pos].item())
                    num_nodes = int(coords.shape[-2])
                    sample_index, node_index = divmod(flat_index, num_nodes)
                    coords_value = coords.detach()
                    if coords_value.ndim == 2:
                        coord = coords_value[node_index]
                    else:
                        coord = coords_value[sample_index, node_index]

                    target_step = self._batch_scalar(
                        batch.get("target_step"), sample_index
                    )
                    target_time = self._batch_scalar(
                        batch.get("target_time"), sample_index
                    )
                    worst_case = {
                        "target_step": int(target_step) if target_step is not None else -1,
                        "target_time_raw": target_time,
                        "target_time_s": (
                            target_time * self.loss_fn.time_scale_to_s
                            if target_time is not None else None
                        ),
                        "node_index": node_index,
                        "coord_mm": [float(v) for v in coord.float().cpu().tolist()],
                        "prediction": float(valid_preds[local_max_pos].item()),
                        "target": float(valid_targets[local_max_pos].item()),
                        "abs_error": local_max_error,
                    }

        avg_loss = total_loss / len(val_loader)
        if all_preds:
            preds = torch.cat(all_preds)
            targs = torch.cat(all_targets)
            metrics = compute_metrics(preds, targs)
        else:
            metrics = {}

        return {
            "val_loss": avg_loss,
            "metrics": metrics,
            "worst_case": worst_case,
        }

    @staticmethod
    def _batch_scalar(value, sample_index: int):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            if flat.numel() == 0:
                return None
            index = min(sample_index, flat.numel() - 1)
            return float(flat[index].item())
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return float(value[min(sample_index, len(value) - 1)])
        return float(value)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int | None = None):
        epochs = epochs or self.config.training.epochs
        self.scheduler.T_max = epochs
        start_epoch = self.current_epoch + 1

        if start_epoch > epochs:
            self.writer.close()
            print(
                f"Checkpoint is already at epoch {self.current_epoch}; "
                f"requested total epochs={epochs}. Nothing to train."
            )
            return

        try:
            for epoch in range(start_epoch, epochs + 1):
                t0 = time.time()
                train_metrics = self.train_epoch(
                    train_loader, epoch, total_epochs=epochs
                )
                train_time = time.time() - t0

                self._log_metrics(train_metrics, epoch, prefix="train")
                self.writer.add_scalar("time/train_epoch", train_time, epoch)

                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, epoch)

                improved = False
                should_stop = False
                if epoch % self.eval_every == 0 or epoch == epochs:
                    val_metrics = self.validate_epoch(val_loader, epoch=epoch)
                    self._log_metrics(val_metrics, epoch, prefix="val")

                    val_loss = val_metrics["val_loss"]
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.epochs_no_improve = 0
                        improved = True
                    else:
                        self.epochs_no_improve += 1

                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4e} | "
                        f"Val: {val_loss:.4e} | "
                        f"Best: {self.best_val_loss:.4e} @ epoch {self.best_epoch} | "
                        f"Time: {train_time:.1f}s | "
                        f"{train_metrics['seconds_per_batch']:.1f}s/b | "
                        f"VRAM: {train_metrics['peak_vram_gb']:.1f}GB"
                    )

                    should_stop = (
                        self.early_stopping_patience > 0
                        and self.epochs_no_improve >= self.early_stopping_patience
                    )
                else:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4e} | "
                        f"Time: {train_time:.1f}s | "
                        f"{train_metrics['seconds_per_batch']:.1f}s/b | "
                        f"VRAM: {train_metrics['peak_vram_gb']:.1f}GB"
                    )

                self.scheduler.step()
                self.current_epoch = epoch

                if improved:
                    self.save_checkpoint("best_model.pt")
                self.save_checkpoint("last_model.pt")
                if epoch % self.save_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

                if should_stop:
                    print(f"Early stopping after {epoch} epochs.")
                    break
        except KeyboardInterrupt:
            self.save_checkpoint("interrupted_model.pt")
            print(
                "\nTraining interrupted. Saved recovery checkpoint to "
                f"{self.log_dir / 'interrupted_model.pt'}"
            )
            raise
        finally:
            self.writer.close()

        print(f"Training complete. Best val_loss={self.best_val_loss:.4e} at epoch {self.best_epoch}")

    def _log_metrics(self, metrics: dict, epoch: int, prefix: str):
        for key, value in metrics.items():
            if key == "metrics":
                for mk, mv in value.items():
                    self.writer.add_scalar(f"{prefix}/{mk}", mv, epoch)
            elif isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(f"{prefix}/{key}", value, epoch)

    def save_checkpoint(self, filename: str):
        path = self.log_dir / filename
        temp_path = path.with_suffix(path.suffix + ".tmp")
        state = {
            "checkpoint_version": 2,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "epochs_no_improve": self.epochs_no_improve,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "config": self.config,
        }
        torch.save(state, temp_path)
        temp_path.replace(path)

    def load_checkpoint(self, path: str | Path):
        path = Path(path)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        fallback_epoch = self.best_epoch
        epoch_match = re.search(r"checkpoint_epoch_(\d+)", path.stem)
        if epoch_match:
            fallback_epoch = int(epoch_match.group(1))
        self.current_epoch = int(ckpt.get("epoch", fallback_epoch))
        self.global_step = int(ckpt.get("global_step", 0))
        self.epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        cuda_rng_state = ckpt.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])

        return self.current_epoch

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
