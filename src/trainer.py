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
from .utils.metrics import compute_binary_detection_metrics, compute_metrics

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
        self.hot_checkpoint_enabled = bool(getattr(
            config.logging, "save_hot_checkpoint", True
        ))
        self.hot_checkpoint_metric = str(getattr(
            config.logging, "hot_checkpoint_metric", "TempF1AboveSolidus"
        ))
        self.hot_checkpoint_mode = str(getattr(
            config.logging, "hot_checkpoint_mode", "max"
        )).lower()
        if self.hot_checkpoint_mode not in {"min", "max"}:
            raise ValueError("logging.hot_checkpoint_mode must be 'min' or 'max'.")
        self.best_hot_score = (
            -float("inf") if self.hot_checkpoint_mode == "max" else float("inf")
        )
        self.best_hot_epoch = 0
        self.epochs_no_improve = 0
        self.global_step = 0
        self.current_epoch = 0
        self.use_base_for_field_losses = bool(getattr(
            config.loss, "use_base_for_field_losses", False
        ))

    def _field_loss_prediction(self, output: dict) -> torch.Tensor:
        if not self.use_base_for_field_losses:
            return output["T_pred"]
        return output.get("T_base", output["T_pred"])

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
        if hasattr(self.model, "set_training_epoch"):
            self.model.set_training_epoch(epoch)
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
            laser_pos = batch.get("target_laser_pos", None)
            if laser_pos is None:
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
                field_pred = self._field_loss_prediction(output)
                total, components = self.loss_fn.forward(
                    pred=field_pred, target=target, prev_temp=prev_temp,
                    coords=coords, edge_index=edge_index, boundary=boundary,
                    dt=dt, laser_pos=laser_pos, mask=mask,
                    is_initial=is_initial,
                    hotspot_logit=output.get("hotspot_logit"),
                    hotspot_delta=output.get("hotspot_delta"),
                    hotspot_specialist_temp=output.get("hotspot_specialist_temp"),
                    hotspot_specialist_gate=output.get("hotspot_specialist_gate"),
                    process_gate=output.get("process_gate"),
                    laser_arrival_gate=output.get("laser_arrival_gate"),
                    laser_path_post_arrival_gate=output.get("laser_path_post_arrival_gate"),
                    laser_endpoint_gate=output.get("laser_endpoint_gate"),
                    neighbor_hot_gate=output.get("neighbor_hot_gate"),
                    final_pred=T_pred,
                    laser_residual_prior_gate=output.get("laser_residual_prior_gate"),
                    laser_residual_learned_gate=output.get("laser_residual_learned_gate"),
                    laser_residual_gate=output.get("laser_residual_gate"),
                    laser_residual_delta=output.get("laser_residual_delta"),
                    laser_residual_boost=output.get("laser_residual_boost"),
                    cold_to_hot_logit=output.get("cold_to_hot_logit"),
                    cold_to_hot_gate=output.get("cold_to_hot_gate"),
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
            if "HotCls" in components:
                postfix["HotCls"] = f"{components['HotCls'].item():.3e}"
            if "HotDelta" in components:
                postfix["HotDelta"] = f"{components['HotDelta'].item():.3e}"
            if "HotSpec" in components:
                postfix["HotSpec"] = f"{components['HotSpec'].item():.3e}"
            if "HotSpecProcess" in components:
                postfix["HotSpecProcess"] = f"{components['HotSpecProcess'].item():.3e}"
            if "HotFinal" in components:
                postfix["HotFinal"] = f"{components['HotFinal'].item():.3e}"
            if "ColdFalseHot" in components:
                postfix["ColdFalseHot"] = f"{components['ColdFalseHot'].item():.3e}"
            if "SpecialistFalseHot" in components:
                postfix["SpecialistFalseHot"] = f"{components['SpecialistFalseHot'].item():.3e}"
            if "LaserResidual" in components:
                postfix["LaserResidual"] = f"{components['LaserResidual'].item():.3e}"
            if "LaserResidualDelta" in components:
                postfix["LaserResidualDelta"] = f"{components['LaserResidualDelta'].item():.3e}"
            if "LaserResidualGate" in components:
                postfix["LaserResidualGate"] = f"{components['LaserResidualGate'].item():.3e}"
            if "ColdToHotCls" in components:
                postfix["ColdToHotCls"] = f"{components['ColdToHotCls'].item():.3e}"
            if "FinalT" in components:
                postfix["FinalT"] = f"{components['FinalT'].item():.3e}"
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
            "specialist_blend_weight": float(getattr(
                self.model, "hotspot_specialist_blend_weight", 1.0
            )),
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader,
                       epoch: int | None = None) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        all_hot_probs, all_hot_targets = [], []
        all_spec_gates, all_process_gates, all_neighbor_gates = [], [], []
        all_cold_to_hot_gates, all_cold_to_hot_targets = [], []
        all_target_heat_gates, all_sweep_heat_gates, all_arrival_gates = [], [], []
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
            laser_pos = batch.get("target_laser_pos", None)
            if laser_pos is None:
                laser_pos = getattr(graph_seq[-1], "laser_pos",
                                    torch.zeros(3, device=self.device))

            with autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                output = self.model(graph_seq, dt=dt)
                T_pred = output["T_pred"]
                field_pred = self._field_loss_prediction(output)
                total, _ = self.loss_fn.forward(
                    pred=field_pred, target=target, prev_temp=prev_temp,
                    coords=coords, edge_index=edge_index, boundary=boundary,
                    dt=dt, laser_pos=laser_pos, mask=mask,
                    hotspot_logit=output.get("hotspot_logit"),
                    hotspot_delta=output.get("hotspot_delta"),
                    hotspot_specialist_temp=output.get("hotspot_specialist_temp"),
                    hotspot_specialist_gate=output.get("hotspot_specialist_gate"),
                    process_gate=output.get("process_gate"),
                    laser_arrival_gate=output.get("laser_arrival_gate"),
                    laser_path_post_arrival_gate=output.get("laser_path_post_arrival_gate"),
                    laser_endpoint_gate=output.get("laser_endpoint_gate"),
                    neighbor_hot_gate=output.get("neighbor_hot_gate"),
                    final_pred=T_pred,
                    laser_residual_prior_gate=output.get("laser_residual_prior_gate"),
                    laser_residual_learned_gate=output.get("laser_residual_learned_gate"),
                    laser_residual_gate=output.get("laser_residual_gate"),
                    laser_residual_delta=output.get("laser_residual_delta"),
                    laser_residual_boost=output.get("laser_residual_boost"),
                    cold_to_hot_logit=output.get("cold_to_hot_logit"),
                    cold_to_hot_gate=output.get("cold_to_hot_gate"),
                )
            total_loss += total.item()
            progress.set_postfix({
                "loss": f"{total.item():.3e}",
                "avg": f"{total_loss / (batch_idx + 1):.3e}",
            })

            pred_flat = T_pred.detach().float().reshape(-1)
            target_flat = target.detach().float().reshape(-1)
            hot_prob_flat = None
            hotspot_logit = output.get("hotspot_logit")
            if hotspot_logit is not None:
                hot_prob_flat = torch.sigmoid(
                    hotspot_logit.detach().float()
                ).reshape(-1)
            spec_temp_flat = None
            if output.get("hotspot_specialist_temp") is not None:
                spec_temp_flat = output["hotspot_specialist_temp"].detach().float().reshape(-1)
            spec_gate_flat = None
            spec_raw_gate_flat = None
            if output.get("hotspot_specialist_raw_gate") is not None:
                spec_raw_gate_flat = output["hotspot_specialist_raw_gate"].detach().float().reshape(-1)
                spec_gate_flat = spec_raw_gate_flat
            elif output.get("hotspot_specialist_gate") is not None:
                spec_gate_flat = output["hotspot_specialist_gate"].detach().float().reshape(-1)
            spec_effective_gate_flat = None
            if output.get("hotspot_specialist_gate") is not None:
                spec_effective_gate_flat = output["hotspot_specialist_gate"].detach().float().reshape(-1)
            process_gate_flat = None
            if output.get("process_gate") is not None:
                process_gate_flat = output["process_gate"].detach().float().reshape(-1)
            target_heat_gate_flat = None
            if output.get("laser_target_heat_gate") is not None:
                target_heat_gate_flat = output["laser_target_heat_gate"].detach().float().reshape(-1)
            body_heat_gate_flat = None
            if output.get("laser_body_heat_gate") is not None:
                body_heat_gate_flat = output["laser_body_heat_gate"].detach().float().reshape(-1)
            path_body_heat_gate_flat = None
            if output.get("laser_path_body_heat_gate") is not None:
                path_body_heat_gate_flat = output["laser_path_body_heat_gate"].detach().float().reshape(-1)
            path_wake_gate_flat = None
            if output.get("laser_path_wake_gate") is not None:
                path_wake_gate_flat = output["laser_path_wake_gate"].detach().float().reshape(-1)
            sweep_heat_gate_flat = None
            if output.get("laser_sweep_heat_gate") is not None:
                sweep_heat_gate_flat = output["laser_sweep_heat_gate"].detach().float().reshape(-1)
            arrival_gate_flat = None
            if output.get("laser_arrival_gate") is not None:
                arrival_gate_flat = output["laser_arrival_gate"].detach().float().reshape(-1)
            path_active_gate_flat = None
            if output.get("laser_path_active_gate") is not None:
                path_active_gate_flat = output["laser_path_active_gate"].detach().float().reshape(-1)
            path_pre_arrival_gate_flat = None
            if output.get("laser_path_pre_arrival_gate") is not None:
                path_pre_arrival_gate_flat = output["laser_path_pre_arrival_gate"].detach().float().reshape(-1)
            path_post_arrival_gate_flat = None
            if output.get("laser_path_post_arrival_gate") is not None:
                path_post_arrival_gate_flat = output["laser_path_post_arrival_gate"].detach().float().reshape(-1)
            endpoint_gate_flat = None
            if output.get("laser_endpoint_gate") is not None:
                endpoint_gate_flat = output["laser_endpoint_gate"].detach().float().reshape(-1)
            path_program_track_gate_flat = None
            if output.get("laser_path_program_track_gate") is not None:
                path_program_track_gate_flat = output["laser_path_program_track_gate"].detach().float().reshape(-1)
            path_elapsed_gate_flat = None
            if output.get("laser_path_elapsed_gate") is not None:
                path_elapsed_gate_flat = output["laser_path_elapsed_gate"].detach().float().reshape(-1)
            path_time_until_gate_flat = None
            if output.get("laser_path_time_until_gate") is not None:
                path_time_until_gate_flat = output["laser_path_time_until_gate"].detach().float().reshape(-1)
            path_scanned_gate_flat = None
            if output.get("laser_path_scanned_gate") is not None:
                path_scanned_gate_flat = output["laser_path_scanned_gate"].detach().float().reshape(-1)
            path_cooling_tail_gate_flat = None
            if output.get("laser_path_cooling_tail_gate") is not None:
                path_cooling_tail_gate_flat = output["laser_path_cooling_tail_gate"].detach().float().reshape(-1)
            neighbor_gate_flat = None
            if output.get("neighbor_hot_gate") is not None:
                neighbor_gate_flat = output["neighbor_hot_gate"].detach().float().reshape(-1)
            prior_temp_flat = None
            if output.get("hotspot_laser_prior_temp") is not None:
                prior_temp_flat = output["hotspot_laser_prior_temp"].detach().float().reshape(-1)
            prior_gate_flat = None
            if output.get("hotspot_laser_prior_gate") is not None:
                prior_gate_flat = output["hotspot_laser_prior_gate"].detach().float().reshape(-1)
            residual_prior_gate_flat = None
            if output.get("laser_residual_prior_gate") is not None:
                residual_prior_gate_flat = output["laser_residual_prior_gate"].detach().float().reshape(-1)
            residual_learned_gate_flat = None
            if output.get("laser_residual_learned_gate") is not None:
                residual_learned_gate_flat = output["laser_residual_learned_gate"].detach().float().reshape(-1)
            residual_control_gate_flat = None
            if output.get("laser_residual_control_gate") is not None:
                residual_control_gate_flat = output["laser_residual_control_gate"].detach().float().reshape(-1)
            residual_cold_start_gate_flat = None
            if output.get("laser_residual_cold_start_gate") is not None:
                residual_cold_start_gate_flat = output["laser_residual_cold_start_gate"].detach().float().reshape(-1)
            residual_gate_flat = None
            if output.get("laser_residual_gate") is not None:
                residual_gate_flat = output["laser_residual_gate"].detach().float().reshape(-1)
            residual_delta_flat = None
            if output.get("laser_residual_delta") is not None:
                residual_delta_flat = output["laser_residual_delta"].detach().float().reshape(-1)
            residual_boost_flat = None
            if output.get("laser_residual_boost") is not None:
                residual_boost_flat = output["laser_residual_boost"].detach().float().reshape(-1)
            residual_post_rescue_cap_gate_flat = None
            if output.get("laser_residual_post_rescue_cap_gate") is not None:
                residual_post_rescue_cap_gate_flat = output["laser_residual_post_rescue_cap_gate"].detach().float().reshape(-1)
            cold_to_hot_gate_flat = None
            if output.get("cold_to_hot_gate") is not None:
                cold_to_hot_gate_flat = output["cold_to_hot_gate"].detach().float().reshape(-1)
            valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
            if hot_prob_flat is not None:
                valid_mask &= torch.isfinite(hot_prob_flat)
            if spec_temp_flat is not None:
                valid_mask &= torch.isfinite(spec_temp_flat)
            if spec_gate_flat is not None:
                valid_mask &= torch.isfinite(spec_gate_flat)
            if spec_effective_gate_flat is not None:
                valid_mask &= torch.isfinite(spec_effective_gate_flat)
            if process_gate_flat is not None:
                valid_mask &= torch.isfinite(process_gate_flat)
            if target_heat_gate_flat is not None:
                valid_mask &= torch.isfinite(target_heat_gate_flat)
            if sweep_heat_gate_flat is not None:
                valid_mask &= torch.isfinite(sweep_heat_gate_flat)
            if arrival_gate_flat is not None:
                valid_mask &= torch.isfinite(arrival_gate_flat)
            if path_active_gate_flat is not None:
                valid_mask &= torch.isfinite(path_active_gate_flat)
            if path_pre_arrival_gate_flat is not None:
                valid_mask &= torch.isfinite(path_pre_arrival_gate_flat)
            if path_post_arrival_gate_flat is not None:
                valid_mask &= torch.isfinite(path_post_arrival_gate_flat)
            if endpoint_gate_flat is not None:
                valid_mask &= torch.isfinite(endpoint_gate_flat)
            if path_program_track_gate_flat is not None:
                valid_mask &= torch.isfinite(path_program_track_gate_flat)
            if path_elapsed_gate_flat is not None:
                valid_mask &= torch.isfinite(path_elapsed_gate_flat)
            if path_time_until_gate_flat is not None:
                valid_mask &= torch.isfinite(path_time_until_gate_flat)
            if neighbor_gate_flat is not None:
                valid_mask &= torch.isfinite(neighbor_gate_flat)
            if prior_temp_flat is not None:
                valid_mask &= torch.isfinite(prior_temp_flat)
            if prior_gate_flat is not None:
                valid_mask &= torch.isfinite(prior_gate_flat)
            if residual_prior_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_prior_gate_flat)
            if residual_learned_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_learned_gate_flat)
            if residual_control_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_control_gate_flat)
            if residual_cold_start_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_cold_start_gate_flat)
            if residual_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_gate_flat)
            if residual_delta_flat is not None:
                valid_mask &= torch.isfinite(residual_delta_flat)
            if residual_boost_flat is not None:
                valid_mask &= torch.isfinite(residual_boost_flat)
            if residual_post_rescue_cap_gate_flat is not None:
                valid_mask &= torch.isfinite(residual_post_rescue_cap_gate_flat)
            if cold_to_hot_gate_flat is not None:
                valid_mask &= torch.isfinite(cold_to_hot_gate_flat)

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
                if spec_gate_flat is not None:
                    all_spec_gates.append(spec_gate_flat[valid_mask].cpu())
                if process_gate_flat is not None:
                    all_process_gates.append(process_gate_flat[valid_mask].cpu())
                if target_heat_gate_flat is not None:
                    all_target_heat_gates.append(target_heat_gate_flat[valid_mask].cpu())
                if sweep_heat_gate_flat is not None:
                    all_sweep_heat_gates.append(sweep_heat_gate_flat[valid_mask].cpu())
                if arrival_gate_flat is not None:
                    all_arrival_gates.append(arrival_gate_flat[valid_mask].cpu())
                if neighbor_gate_flat is not None:
                    all_neighbor_gates.append(neighbor_gate_flat[valid_mask].cpu())
                if hot_prob_flat is not None:
                    all_hot_probs.append(hot_prob_flat[valid_mask].cpu())
                    all_hot_targets.append(valid_targets.cpu())
                if cold_to_hot_gate_flat is not None and prev_temp is not None:
                    prev_flat = prev_temp.detach().float().reshape(-1)
                    cold_to_hot_target = self._cold_to_hot_target(
                        target_flat,
                        prev_flat,
                    )
                    all_cold_to_hot_gates.append(cold_to_hot_gate_flat[valid_mask].cpu())
                    all_cold_to_hot_targets.append(cold_to_hot_target[valid_mask].cpu())

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
                    if hot_prob_flat is not None:
                        worst_case["hotspot_probability"] = float(
                            hot_prob_flat[flat_index].item()
                        )
                    if spec_temp_flat is not None:
                        worst_case["hotspot_specialist_temp"] = float(
                            spec_temp_flat[flat_index].item()
                        )
                    if spec_gate_flat is not None:
                        worst_case["hotspot_specialist_raw_gate"] = float(
                            spec_gate_flat[flat_index].item()
                        )
                    if spec_effective_gate_flat is not None:
                        worst_case["hotspot_specialist_gate"] = float(
                            spec_effective_gate_flat[flat_index].item()
                        )
                    if process_gate_flat is not None:
                        worst_case["process_gate"] = float(
                            process_gate_flat[flat_index].item()
                        )
                    if target_heat_gate_flat is not None:
                        worst_case["laser_target_heat_gate"] = float(
                            target_heat_gate_flat[flat_index].item()
                        )
                    if body_heat_gate_flat is not None:
                        worst_case["laser_body_heat_gate"] = float(
                            body_heat_gate_flat[flat_index].item()
                        )
                    if path_body_heat_gate_flat is not None:
                        worst_case["laser_path_body_heat_gate"] = float(
                            path_body_heat_gate_flat[flat_index].item()
                        )
                    if path_wake_gate_flat is not None:
                        worst_case["laser_path_wake_gate"] = float(
                            path_wake_gate_flat[flat_index].item()
                        )
                    if sweep_heat_gate_flat is not None:
                        worst_case["laser_sweep_heat_gate"] = float(
                            sweep_heat_gate_flat[flat_index].item()
                        )
                    if arrival_gate_flat is not None:
                        worst_case["laser_arrival_gate"] = float(
                            arrival_gate_flat[flat_index].item()
                        )
                    if path_active_gate_flat is not None:
                        worst_case["laser_path_active_gate"] = float(
                            path_active_gate_flat[flat_index].item()
                        )
                    if path_pre_arrival_gate_flat is not None:
                        worst_case["laser_path_pre_arrival_gate"] = float(
                            path_pre_arrival_gate_flat[flat_index].item()
                        )
                    if path_post_arrival_gate_flat is not None:
                        worst_case["laser_path_post_arrival_gate"] = float(
                            path_post_arrival_gate_flat[flat_index].item()
                        )
                    if endpoint_gate_flat is not None:
                        worst_case["laser_endpoint_gate"] = float(
                            endpoint_gate_flat[flat_index].item()
                        )
                    if path_program_track_gate_flat is not None:
                        worst_case["laser_path_program_track_gate"] = float(
                            path_program_track_gate_flat[flat_index].item()
                        )
                    if path_elapsed_gate_flat is not None:
                        worst_case["laser_path_elapsed_gate"] = float(
                            path_elapsed_gate_flat[flat_index].item()
                        )
                    if path_time_until_gate_flat is not None:
                        worst_case["laser_path_time_until_gate"] = float(
                            path_time_until_gate_flat[flat_index].item()
                        )
                    if path_scanned_gate_flat is not None:
                        worst_case["laser_path_scanned_gate"] = float(
                            path_scanned_gate_flat[flat_index].item()
                        )
                    if path_cooling_tail_gate_flat is not None:
                        worst_case["laser_path_cooling_tail_gate"] = float(
                            path_cooling_tail_gate_flat[flat_index].item()
                        )
                    if neighbor_gate_flat is not None:
                        worst_case["neighbor_hot_gate"] = float(
                            neighbor_gate_flat[flat_index].item()
                        )
                    if prior_temp_flat is not None:
                        worst_case["hotspot_laser_prior_temp"] = float(
                            prior_temp_flat[flat_index].item()
                        )
                    if prior_gate_flat is not None:
                        worst_case["hotspot_laser_prior_gate"] = float(
                            prior_gate_flat[flat_index].item()
                        )
                    if residual_prior_gate_flat is not None:
                        worst_case["laser_residual_prior_gate"] = float(
                            residual_prior_gate_flat[flat_index].item()
                        )
                    if residual_learned_gate_flat is not None:
                        worst_case["laser_residual_learned_gate"] = float(
                            residual_learned_gate_flat[flat_index].item()
                        )
                    if residual_control_gate_flat is not None:
                        worst_case["laser_residual_control_gate"] = float(
                            residual_control_gate_flat[flat_index].item()
                        )
                    if residual_cold_start_gate_flat is not None:
                        worst_case["laser_residual_cold_start_gate"] = float(
                            residual_cold_start_gate_flat[flat_index].item()
                        )
                    if residual_gate_flat is not None:
                        worst_case["laser_residual_gate"] = float(
                            residual_gate_flat[flat_index].item()
                        )
                    if residual_delta_flat is not None:
                        worst_case["laser_residual_delta"] = float(
                            residual_delta_flat[flat_index].item()
                        )
                    if residual_boost_flat is not None:
                        worst_case["laser_residual_boost"] = float(
                            residual_boost_flat[flat_index].item()
                        )
                    if residual_post_rescue_cap_gate_flat is not None:
                        worst_case["laser_residual_post_rescue_cap_gate"] = float(
                            residual_post_rescue_cap_gate_flat[flat_index].item()
                        )
                    if cold_to_hot_gate_flat is not None:
                        worst_case["cold_to_hot_gate"] = float(
                            cold_to_hot_gate_flat[flat_index].item()
                        )

        avg_loss = total_loss / len(val_loader)
        if all_preds:
            preds = torch.cat(all_preds)
            targs = torch.cat(all_targets)
            metrics = compute_metrics(preds, targs)
            temp_det = compute_binary_detection_metrics(
                preds,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=self.loss_fn.hot_cls_threshold,
            )
            metrics.update({
                "TempRecallAboveSolidus": temp_det["recall"],
                "TempPrecisionAboveSolidus": temp_det["precision"],
                "TempF1AboveSolidus": temp_det["f1"],
            })
            if all_hot_probs:
                hot_probs = torch.cat(all_hot_probs)
                hot_targs = torch.cat(all_hot_targets)
                cls_det = compute_binary_detection_metrics(
                    hot_probs,
                    hot_targs,
                    target_threshold=self.loss_fn.hot_cls_threshold,
                    score_threshold=0.5,
                )
                metrics.update({
                    "HotClsRecallAboveSolidus": cls_det["recall"],
                    "HotClsPrecisionAboveSolidus": cls_det["precision"],
                    "HotClsF1AboveSolidus": cls_det["f1"],
                    "HotClsTP": cls_det["true_positive"],
                    "HotClsFP": cls_det["false_positive"],
                    "HotClsFN": cls_det["false_negative"],
                })
            if all_cold_to_hot_gates:
                cold_scores = torch.cat(all_cold_to_hot_gates)
                cold_targets = torch.cat(all_cold_to_hot_targets)
                cold_det = compute_binary_detection_metrics(
                    cold_scores,
                    cold_targets,
                    target_threshold=0.5,
                    score_threshold=0.5,
                )
                metrics.update({
                    "ColdToHotRecall": cold_det["recall"],
                    "ColdToHotPrecision": cold_det["precision"],
                    "ColdToHotF1": cold_det["f1"],
                    "ColdToHotTP": cold_det["true_positive"],
                    "ColdToHotFP": cold_det["false_positive"],
                    "ColdToHotFN": cold_det["false_negative"],
                })
            self._append_gate_metrics(
                metrics,
                "SpecGate",
                all_spec_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
            self._append_gate_metrics(
                metrics,
                "ProcessGate",
                all_process_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
            self._append_gate_metrics(
                metrics,
                "TargetHeatGate",
                all_target_heat_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
            self._append_gate_metrics(
                metrics,
                "SweepHeatGate",
                all_sweep_heat_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
            self._append_gate_metrics(
                metrics,
                "ArrivalGate",
                all_arrival_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
            self._append_gate_metrics(
                metrics,
                "NeighborGate",
                all_neighbor_gates,
                targs,
                target_threshold=self.loss_fn.hot_cls_threshold,
                score_threshold=0.1,
            )
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

    def _cold_to_hot_target(
            self,
            target: torch.Tensor,
            prev_temp: torch.Tensor,
            ) -> torch.Tensor:
        target = target.float().reshape(-1)
        prev_temp = prev_temp.float().reshape(-1)
        label = target >= float(self.loss_fn.cold_to_hot_threshold)
        prev_max = float(self.loss_fn.cold_to_hot_prev_max_temp)
        if prev_max > 0:
            label &= prev_temp <= prev_max
        min_gap = float(self.loss_fn.cold_to_hot_min_target_gap)
        if min_gap > 0:
            label &= (target - prev_temp).clamp_min(0.0) >= min_gap
        return label.float()

    @staticmethod
    def _append_gate_metrics(metrics: dict, name: str, scores_list: list,
                             targets: torch.Tensor, *, target_threshold: float,
                             score_threshold: float) -> None:
        if not scores_list:
            return
        scores = torch.cat(scores_list).float().reshape(-1)
        targets = targets.float().reshape(-1)
        if scores.numel() != targets.numel():
            return
        det = compute_binary_detection_metrics(
            scores,
            targets,
            target_threshold=target_threshold,
            score_threshold=score_threshold,
        )
        metrics.update({
            f"{name}RecallAt{score_threshold:g}": det["recall"],
            f"{name}PrecisionAt{score_threshold:g}": det["precision"],
            f"{name}F1At{score_threshold:g}": det["f1"],
        })

        hot_mask = targets >= target_threshold
        normal_mask = ~hot_mask
        if hot_mask.any():
            hot_scores = scores[hot_mask].detach().cpu().numpy()
            metrics[f"{name}HotP50"] = float(np.quantile(hot_scores, 0.50))
            metrics[f"{name}HotP90"] = float(np.quantile(hot_scores, 0.90))
        if normal_mask.any():
            normal_scores = scores[normal_mask].detach().cpu().numpy()
            metrics[f"{name}NormalP99"] = float(np.quantile(normal_scores, 0.99))

    def _select_metric_value(self, val_metrics: dict, metric_name: str) -> float | None:
        if metric_name == "val_loss":
            return float(val_metrics["val_loss"])
        metrics = val_metrics.get("metrics", {})
        value = metrics.get(metric_name)
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def _is_better_hot_score(self, score: float) -> bool:
        if self.hot_checkpoint_mode == "max":
            return score > self.best_hot_score
        return score < self.best_hot_score

    def _hot_metric_summary(self, val_metrics: dict) -> str:
        metrics = val_metrics.get("metrics", {})
        pieces = []
        for key, label in [
                ("TempRecallAboveSolidus", "Rec"),
                ("TempPrecisionAboveSolidus", "Prec"),
                ("TempF1AboveSolidus", "F1"),
                ("AbsErrorP99", "P99"),
                ("MaxError", "MaxErr"),
        ]:
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                pieces.append(f"{label}: {float(value):.3g}")
        return " | ".join(pieces)

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
                hot_improved = False
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

                    hot_score = self._select_metric_value(
                        val_metrics, self.hot_checkpoint_metric
                    )
                    if (
                            self.hot_checkpoint_enabled
                            and hot_score is not None
                            and self._is_better_hot_score(hot_score)):
                        self.best_hot_score = hot_score
                        self.best_hot_epoch = epoch
                        hot_improved = True

                    hot_status = ""
                    if self.hot_checkpoint_enabled and hot_score is not None:
                        hot_status = (
                            f" | HotBest({self.hot_checkpoint_metric}): "
                            f"{self.best_hot_score:.4e} @ epoch {self.best_hot_epoch}"
                        )
                    hot_metrics = self._hot_metric_summary(val_metrics)
                    if hot_metrics:
                        hot_status += f" | {hot_metrics}"

                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4e} | "
                        f"Val: {val_loss:.4e} | "
                        f"Best: {self.best_val_loss:.4e} @ epoch {self.best_epoch} | "
                        f"SpecBlend: {train_metrics['specialist_blend_weight']:.2f} | "
                        f"Time: {train_time:.1f}s | "
                        f"{train_metrics['seconds_per_batch']:.1f}s/b | "
                        f"VRAM: {train_metrics['peak_vram_gb']:.1f}GB"
                        f"{hot_status}"
                    )

                    should_stop = (
                        self.early_stopping_patience > 0
                        and self.epochs_no_improve >= self.early_stopping_patience
                    )
                else:
                    print(
                        f"Epoch {epoch:4d}/{epochs} | "
                        f"Train: {train_metrics['loss']:.4e} | "
                        f"SpecBlend: {train_metrics['specialist_blend_weight']:.2f} | "
                        f"Time: {train_time:.1f}s | "
                        f"{train_metrics['seconds_per_batch']:.1f}s/b | "
                        f"VRAM: {train_metrics['peak_vram_gb']:.1f}GB"
                    )

                self.scheduler.step()
                self.current_epoch = epoch

                if improved:
                    self.save_checkpoint("best_model.pt")
                if hot_improved:
                    self.save_checkpoint("best_hot_model.pt")
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
        if self.best_hot_epoch > 0:
            print(
                f"Best {self.hot_checkpoint_metric}={self.best_hot_score:.4e} "
                f"at epoch {self.best_hot_epoch}"
            )

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
            "hot_checkpoint_metric": self.hot_checkpoint_metric,
            "hot_checkpoint_mode": self.hot_checkpoint_mode,
            "best_hot_score": self.best_hot_score,
            "best_hot_epoch": self.best_hot_epoch,
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
        self.best_hot_score = ckpt.get("best_hot_score", self.best_hot_score)
        self.best_hot_epoch = ckpt.get("best_hot_epoch", 0)
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
