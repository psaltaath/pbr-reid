"""Configuration loading (CLI) and pretty-printing."""

import sys

from rich import box
from rich.console import Console
from rich.table import Table

from config import ConfigManager, UnifiedConfig


def _split_argv(args):
    """Return (positional names, extra config files, remaining args for overrides)."""
    names, files, rest = [], [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--config-file", "-f"):
            files.extend(args[i + 1].split(","))
            i += 2
            continue
        if a.startswith("--config-file="):
            files.extend(a.split("=", 1)[1].split(","))
        elif a.startswith("--"):
            rest.append(a)
            if "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("--"):
                rest.append(args[i + 1])
                i += 1
        else:
            names.append(a)
        i += 1
    return names, files, rest


def get_configuration(argv=None) -> UnifiedConfig:
    """Parse ``train.py <name> [<name> ...] [--config-file a.yaml,b.yaml] [--section.key value ...]``.

    Named configs are resolved under ``configs/`` (e.g. ``market1501``,
    ``vitl16``, ``ablations/losses/l1_bht``); later sources override earlier ones.
    """
    console = Console()
    manager = ConfigManager()
    args = list(sys.argv[1:] if argv is None else argv)

    if "--list-configs" in args:
        console.print("[bold blue]Available configurations:[/bold blue]")
        for name in manager.list_available_configs():
            console.print(f"  • {name}")
        sys.exit(0)

    names, files, rest = _split_argv(args)
    config = manager.load_config(config_name=names, config_path=files, cli_args=rest)
    console.print(
        f"[bold green]✅ Configuration loaded[/bold green] "
        f"(base + {', '.join(names) or 'no named config'}"
        f"{' + ' + ', '.join(files) if files else ''})"
    )
    return config


def _kv_table(title, rows):
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    for k, v in rows:
        table.add_row(k, str(v))
    return table


def print_configuration(config: UnifiedConfig):
    console = Console()
    T, M, D, L, E, S = config.training, config.model, config.data, config.loss, config.evaluation, config.system
    console.print("\n[bold blue]🔧 Configuration[/bold blue]")

    console.print(
        _kv_table(
            "Model",
            [
                ("Backbone", f"{M.backbone} ({M.model_tag}, {M.pretrain_dataset})" if M.backbone == "dinov3" else M.backbone),
                ("Prototype EMA decay", M.ema_decay),
                ("Prototype init gain", M.prototype_init_gain),
            ],
        )
    )
    console.print(
        _kv_table(
            "Data",
            [
                ("Dataset", D.dataset_name),
                ("Root", D.root_dir),
                ("Train size (H x W)", f"{D.size_train[0]}x{D.size_train[1]}"),
                ("Test size (H x W)", f"{D.size_test[0]}x{D.size_test[1]}"),
                ("Sampler", f"{D.sampler} (B={D.train_batch_size}, N_k={D.num_instance})"),
                ("Augmentation", f"flip p={D.horizontal_flip_prob}, pad {D.padding} + crop, erase p={D.random_erase_prob}"),
            ],
        )
    )
    console.print(
        _kv_table(
            "Training",
            [
                ("Epochs / warmup", f"{T.num_epochs} / {T.prototype_warmup_epochs}"),
                ("Warmup prototype LR", f"{T.warmup_lr:g} → {T.warmup_min_lr:g} (cosine)"),
                ("Dual optimizers", T.use_dual_optimizers),
                ("Encoder LR", f"{T.encoder_lr:g} → {T.encoder_min_lr:g} ({T.encoder_scheduler})"),
                ("Prototype LR", f"{T.prototype_lr:g} ({T.prototype_scheduler}, step {T.prototype_step_size} ×{T.prototype_gamma})"),
                ("Weight decay", T.weight_decay),
                ("Mixed precision", S.mixed_precision),
                ("Seed / deterministic", f"{S.seed} / {S.deterministic}"),
                ("Eval period", T.eval_period),
            ],
        )
    )
    on = lambda b: "✅" if b else "❌"  # noqa: E731
    console.print(
        _kv_table(
            "Objective",
            [
                ("L_fpt", f"{on(L.enable_fpt)} λ={L.fpt_weight}, m={L.fpt_margin}, γ_fpt={L.fpt_similarity_weight_strength if L.fpt_use_similarity_weighting else 'off'}"),
                ("L_bht", f"{on(L.enable_bht)} λ={L.bht_weight}, m={L.bht_margin}"),
                ("L_cen", f"{on(L.enable_center)} λ={L.center_weight}, mean={L.center_mean_type}, γ_sim={L.center_similarity_weight_strength if L.center_use_similarity_weighting else 'off'}, γ_var={L.center_variance_weight_strength if L.center_use_variance_weighting else 'off'}, stop-grad={L.center_stop_gradient_on_prototype}"),
                ("L_norm", f"{on(L.enable_norm)} λ={L.norm_weight}"),
                ("Head", f"{on(L.enable_head)} {L.head} λ={L.head_weight}"),
                ("MixUp", f"{on(T.enable_mixup)} mode={T.mixup_mode}, α={T.mixup_alpha}, p={T.mixup_prob}"),
                ("BRM", f"{on(T.enable_brm and T.enable_mixup)} τ={T.brm_tau}, m={T.brm_margin}, β={T.brm_beta}, λ_BRM={T.brm_lambda}"),
                ("Adaptive separation", f"{on(T.enable_adaptive_separation and T.enable_brm)} γ_sep={T.sep_gamma}, EMA={T.sep_ema_decay}, w={T.sep_weight}, scope={T.sep_pair_scope}"),
            ],
        )
    )
    console.print(
        _kv_table(
            "Evaluation / System",
            [
                ("Max rank", E.max_rank),
                ("Re-ranking", f"{on(E.rerank)} (k1={E.rerank_k1}, k2={E.rerank_k2}, λ={E.rerank_lambda})"),
                ("Device", S.device),
                ("Checkpoint dir", S.checkpoint_dir),
            ],
        )
    )
