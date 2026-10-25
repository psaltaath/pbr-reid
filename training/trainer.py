"""Two-stage trainer (paper Sec. 3.2.3): prototype warmup -> EMA-stabilized joint fine-tuning."""

import os
from typing import Dict, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from config import UnifiedConfig
from evaluation.reid import validate_reid_both, validate_reid_model
from losses import CombinedLoss, FeatureMixUp, batch_hard_triplet, should_apply_mixup
from models.prototype_model import PrototypeReID

from .checkpoint import BestModelTracker, create_training_table, format_loss_string, load_checkpoint, save_checkpoint
from .model_factory import (
    create_dual_finetune_optimizers_and_schedulers,
    create_mixed_precision_scaler,
    create_single_finetune_optimizer_and_scheduler,
)


class Trainer:
    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config
        self.console = Console()
        self.best_tracker = BestModelTracker(metric="mAP")

        self.model: PrototypeReID | None = None
        self.criterion: CombinedLoss | None = None
        self.device: torch.device | None = None
        self.scaler = None

        # Stage 1 uses current_optimizer/scheduler (prototypes only). Stage 2 uses
        # either the dual (encoder_*, prototype_*) pair or current_* (single LR).
        self.current_optimizer: Optimizer | None = None
        self.current_scheduler: LRScheduler | None = None
        self.encoder_optimizer: Optimizer | None = None
        self.encoder_scheduler: LRScheduler | None = None
        self.prototype_optimizer: Optimizer | None = None
        self.prototype_scheduler: LRScheduler | None = None

        self.current_epoch = 0
        self.needs_similarity = False

        T = config.training
        self.mixup = None
        if T.enable_mixup:
            self.mixup = FeatureMixUp(
                alpha=T.mixup_alpha,
                mode=T.mixup_mode,
                neighbor_k=T.mixup_neighbor_k,
                neighbor_temperature=T.mixup_neighbor_temperature,
            )
            what = "BRM" if T.enable_brm else f"MixUp ({T.mixup_mode})"
            self.console.print(f"[cyan]✓ {what} enabled: α={T.mixup_alpha}, p={T.mixup_prob}[/cyan]")

    # ------------------------------------------------------------------ setup
    def setup_training(self, model, criterion, device, warmup_optimizer, warmup_scheduler) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.current_optimizer = warmup_optimizer
        self.current_scheduler = warmup_scheduler
        self.scaler = create_mixed_precision_scaler(self.config, device)

        # Stage 1: frozen backbone, direct (non-EMA) prototype updates.
        self.model.set_backbone_trainable(False)
        self.model.use_ema_prototypes = False

        L, T = self.config.loss, self.config.training
        self.needs_similarity = (
            (L.enable_fpt and L.fpt_use_similarity_weighting)
            or (L.enable_center and L.center_use_similarity_weighting)
            or (T.enable_mixup and (T.enable_brm or T.mixup_mode == "neighbor_hard_target"))
        )

    def switch_to_finetuning(self) -> None:
        """Stage 2: unfreeze the backbone, build the fine-tuning optimizer(s), enable prototype EMA."""
        T = self.config.training
        self.console.print("[yellow]🔄 Prototype warmup done → joint fine-tuning[/yellow]")
        self.model.set_backbone_trainable(True)

        if T.use_dual_optimizers:
            (
                self.encoder_optimizer,
                self.encoder_scheduler,
                self.prototype_optimizer,
                self.prototype_scheduler,
            ) = create_dual_finetune_optimizers_and_schedulers(self.config, self.model)
            self.current_optimizer = self.current_scheduler = None
            self.console.print(
                f"[green]✓ Dual optimizers: encoder lr={T.encoder_lr:.1e} ({T.encoder_scheduler}), "
                f"prototypes lr={T.prototype_lr:.1e} ({T.prototype_scheduler})[/green]"
            )
        else:
            self.current_optimizer, self.current_scheduler = create_single_finetune_optimizer_and_scheduler(
                self.config, self.model
            )
            self.console.print(f"[green]✓ Single optimizer: lr={T.single_lr:.1e} (cosine)[/green]")

        self.model.use_ema_prototypes = True
        self.scaler = create_mixed_precision_scaler(self.config, self.device)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _has_grads(optimizer: Optimizer) -> bool:
        return any(p.grad is not None for g in optimizer.param_groups for p in g["params"])

    def _active_optimizers(self):
        if self.encoder_optimizer is not None:
            return [self.encoder_optimizer, self.prototype_optimizer]
        return [self.current_optimizer]

    def _is_warmup(self, epoch=None) -> bool:
        epoch = self.current_epoch if epoch is None else epoch
        return epoch <= self.config.training.prototype_warmup_epochs

    # ------------------------------------------------------------------ epoch
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.current_epoch = epoch
        phase = "🔥 Warmup" if self._is_warmup(epoch) else "🎯 Fine-tune"
        running_loss, running = 0.0, {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold blue]Loss: {task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]{phase} Epoch {epoch}/{self.config.training.num_epochs}", total=len(train_loader), loss=0.0
            )
            for batch_idx, (imgs, pids, *_) in enumerate(train_loader):
                imgs, pids = imgs.to(self.device), pids.to(self.device)
                for opt in self._active_optimizers():
                    opt.zero_grad()

                if self.scaler is not None:
                    with torch.autocast("cuda"):
                        loss, loss_dict = self._forward_pass(imgs, pids)
                    self.scaler.scale(loss).backward()
                    for opt in self._active_optimizers():
                        if self._has_grads(opt):  # GradScaler asserts on grad-less optimizers
                            self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss, loss_dict = self._forward_pass(imgs, pids)
                    loss.backward()
                    for opt in self._active_optimizers():
                        opt.step()

                self.model.update_prototypes()
                running_loss += loss.item()
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v
                progress.update(task, advance=1, loss=loss.item())
                if batch_idx % self.config.training.log_interval == 0:
                    progress.console.print(
                        f"    Batch {batch_idx}/{len(train_loader)}, {format_loss_string(loss.item(), loss_dict)}"
                    )

        n = max(1, len(train_loader))
        return running_loss / n, {k: v / n for k, v in running.items()}

    # ------------------------------------------------------------------ step
    def _forward_pass(self, imgs: torch.Tensor, pids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One training step: Eq. 1 with the stage-dependent terms.

        Warmup:      L_fpt + L_cen + L_norm (prototypes only).
        Fine-tuning: + L_bht, MixUp / BRM (+ adaptive separation).
        """
        T, L = self.config.training, self.config.loss
        is_warmup = self._is_warmup()

        features, unnorm_features = self.model(imgs, return_unnormalized=True)
        prototypes = self.model.prototypes
        unnorm_prototypes = self.model.unnormalized_prototypes if L.enable_norm else None

        # S is recomputed every step during fine-tuning (not in warmup, where the
        # prototypes are still settling and similarity-based terms stay off).
        if self.needs_similarity and not is_warmup:
            self.model.compute_prototypes_similarity()
        S = self.model.prototypes_similarity

        use_mixup = self.mixup is not None and should_apply_mixup(self.config, is_warmup, self.model.training)
        brm_loss, loss, loss_dict = None, None, None

        if use_mixup and T.enable_brm:
            pairs = self.mixup.mix_features_batch_with_boundary(features, pids, prototypes, S, tau=T.brm_tau)
            if isinstance(pairs, dict):
                # BRM engaged: clean terms on the original batch + lambda_BRM * L_BRM.
                brm_loss = self.mixup.compute_brm_loss(features, prototypes, pairs, margin=T.brm_margin, beta=T.brm_beta)
                loss, loss_dict = self.criterion(
                    features, prototypes, pids,
                    unnorm_features=unnorm_features, unnorm_prototypes=unnorm_prototypes,
                    prototypes_similarity=S, brm_loss=brm_loss, is_warmup_phase=is_warmup,
                )
                loss_dict["brm_pairs"] = float(pairs["idx_i"].numel())
            else:
                # No confusable pair in the batch: fall back to standard MixUp.
                loss, loss_dict = self._mixup_criterion(pairs, features, pids, prototypes, S, unnorm_features, unnorm_prototypes, is_warmup)
        elif use_mixup:
            if T.mixup_mode == "neighbor_hard_target":
                mixed = self.mixup.mix_features_with_neighbors(features, pids, prototypes, S)
            else:
                mixed = self.mixup.mix_features(features, pids)
            loss, loss_dict = self._mixup_criterion(mixed, features, pids, prototypes, S, unnorm_features, unnorm_prototypes, is_warmup)
        else:
            loss, loss_dict = self.criterion(
                features, prototypes, pids,
                unnorm_features=unnorm_features, unnorm_prototypes=unnorm_prototypes,
                prototypes_similarity=S, is_warmup_phase=is_warmup,
            )

        # Adaptive-margin prototype separation (Eq. 13, second term), fine-tuning only.
        sep = getattr(self.criterion, "adaptive_sep", None)
        if sep is not None and not is_warmup:
            sep.update_spread(features, pids)
            l_sep, n_pairs = sep(prototypes, S, targets=pids)
            if n_pairs > 0:
                loss = loss + T.brm_lambda * T.sep_weight * l_sep
                loss_dict["sep_loss"] = l_sep.item()
                loss_dict["sep_pairs"] = float(n_pairs)

        # L_bht (Eq. 5-6), fine-tuning only.
        if L.enable_bht and not is_warmup:
            l_bht = batch_hard_triplet(features, pids, margin=L.bht_margin)
            loss = loss + L.bht_weight * l_bht
            loss_dict["bht_loss"] = l_bht.item()

        return loss, loss_dict

    def _mixup_criterion(self, mixed, features, pids, prototypes, S, unnorm_features, unnorm_prototypes, is_warmup):
        mixed_features, pids_a, pids_b, lam = mixed
        return self.criterion(
            mixed_features, prototypes, pids_a,
            unnorm_features=unnorm_features, unnorm_prototypes=unnorm_prototypes,
            prototypes_similarity=S, is_mixup=True, targets_b=pids_b, lambda_mix=lam,
            original_features=features, original_targets=pids, is_warmup_phase=is_warmup,
        )

    # ------------------------------------------------------------------ eval
    def should_evaluate(self, epoch: int) -> bool:
        T = self.config.training
        return (epoch % T.eval_period == 0 and epoch > T.prototype_warmup_epochs) or epoch == T.num_epochs

    def evaluate(self, test_loaders: dict) -> Dict[str, dict]:
        return {
            name: validate_reid_model(self.model, q, g, self.device, max_rank=self.config.evaluation.max_rank)
            for name, (q, g) in test_loaders.items()
        }

    # ------------------------------------------------------------------ loop
    def run_training_loop(self, train_loader: DataLoader, test_loaders: dict) -> None:
        T = self.config.training
        self.console.print(Panel.fit(Text("🚀 PBR-ReID training", style="bold blue"), border_style="blue"))
        self.console.print(f"  • Epochs: {T.num_epochs} (warmup {T.prototype_warmup_epochs}), eval every {T.eval_period}")
        self.console.print(f"  • Device: {self.device}, train batches: {len(train_loader)}")
        self.console.print(f"  • Test sets: {', '.join(test_loaders)}\n")

        for epoch in range(1, T.num_epochs + 1):
            if epoch == T.prototype_warmup_epochs + 1:
                self.switch_to_finetuning()

            avg_loss, avg_loss_dict = self.train_epoch(epoch, train_loader)

            if self.encoder_scheduler is not None:
                current_lr = self.encoder_scheduler.get_last_lr()[0]
                proto_lr = self.prototype_scheduler.get_last_lr()[0]
                self.encoder_scheduler.step()
                self.prototype_scheduler.step()
                self.console.print(f"[dim]  LR: encoder={current_lr:.2e}, prototypes={proto_lr:.2e}[/dim]")
            else:
                current_lr = self.current_scheduler.get_last_lr()[0]
                self.current_scheduler.step()

            if self.should_evaluate(epoch):
                all_metrics = self.evaluate(test_loaders)
                first = next(iter(all_metrics.values()))
                mAP, rank1 = first["mAP"], first["R-1"]
                self.console.print(create_training_table(epoch, avg_loss, avg_loss_dict, all_metrics, current_lr))
                if self.best_tracker.update(epoch, mAP, rank1):
                    if self.encoder_optimizer is not None:
                        opt_state = {
                            "encoder": self.encoder_optimizer.state_dict(),
                            "prototype": self.prototype_optimizer.state_dict(),
                        }
                    else:
                        opt_state = self.current_optimizer.state_dict()
                    save_checkpoint(self.model, opt_state, epoch, mAP, rank1, self.config.system.checkpoint_dir)
                    self.console.print(f"[green]✓ New best model saved! mAP: {mAP:.2f}%, Rank-1: {rank1:.2f}%[/green]")
            else:
                self.console.print(
                    f"[blue]Epoch {epoch}/{T.num_epochs} - {format_loss_string(avg_loss, avg_loss_dict)}[/blue]"
                )
            self.console.print()

        self._print_training_summary(test_loaders)

    # ------------------------------------------------------------------ summary
    def _print_training_summary(self, test_loaders: dict) -> None:
        self.console.print(Panel.fit(Text("🎉 Training completed", style="bold green"), border_style="green"))
        best_value, best_epoch = self.best_tracker.get_best_info()
        self.console.print(f"  • Best mAP: {best_value:.2f}% at epoch {best_epoch}")
        self.console.print(f"  • Checkpoints: {self.config.system.checkpoint_dir}")
        if best_epoch == 0:
            return

        self._load_best_checkpoint(best_epoch)
        E = self.config.evaluation
        self.console.print(f"\n[cyan]Final evaluation of the best checkpoint (epoch {best_epoch}):[/cyan]")
        for name, (q, g) in test_loaders.items():
            label = "Test set" if name == "default" else f"Test set ({name})"
            if E.rerank:
                plain, rerank = validate_reid_both(
                    self.model, q, g, self.device, max_rank=E.max_rank,
                    rerank_k1=E.rerank_k1, rerank_k2=E.rerank_k2, rerank_lambda=E.rerank_lambda,
                )
                rows = (("plain", plain), ("re-ranked", rerank))
            else:
                rows = (("plain", validate_reid_model(self.model, q, g, self.device, max_rank=E.max_rank)),)
            self.console.print(f"\n[yellow]{label}:[/yellow]")
            for tag, m in rows:
                self.console.print(
                    f"  [{tag:>9}]  mAP: {m['mAP']:.2f}%  R-1: {m['R-1']:.2f}%  "
                    f"R-5: {m['R-5']:.2f}%  R-10: {m['R-10']:.2f}%  R-20: {m['R-20']:.2f}%"
                )

    def _load_best_checkpoint(self, best_epoch: int) -> None:
        ckpt_dir = self.config.system.checkpoint_dir
        for path in (os.path.join(ckpt_dir, f"checkpoint_epoch_{best_epoch}.pth"), os.path.join(ckpt_dir, "latest.pth")):
            if os.path.exists(path):
                meta = load_checkpoint(path, self.model)
                self.console.print(f"[cyan]Loaded {os.path.basename(path)} (epoch {meta['epoch']}, mAP {meta['mAP']:.2f}%)[/cyan]")
                return
        self.console.print("[yellow]⚠ Best checkpoint not found; evaluating in-memory weights.[/yellow]")
