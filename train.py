#!/usr/bin/env python3
"""PBR-ReID training.

    uv run train.py market1501                              # paper recipe (Table 1)
    uv run train.py veri --system.seed 123                  # any --section.key override
    uv run train.py market1501 vitl16                       # stack named configs (configs/**.yaml)
    uv run train.py market1501 -f configs/ablations/losses/l1_bht.yaml
    uv run train.py --list-configs
"""

import re
import sys
from datetime import datetime
from pathlib import Path

from training import (
    Trainer,
    create_data_loaders,
    create_model_and_losses,
    create_warmup_optimizer_and_scheduler,
    get_configuration,
    print_configuration,
    setup_deterministic_training,
    worker_init_fn,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class _Tee:
    """Duplicate a text stream into a log file (ANSI escapes stripped in the file)."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(_ANSI_RE.sub("", data))
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _tee_console_to_logfile(config) -> None:
    if not config.system.save_logs:
        return
    ckpt_dir = Path(config.system.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(ckpt_dir / "train.log", "a", buffering=1, encoding="utf-8")
    log_fh.write(f"\n===== run started {datetime.now().isoformat()} | argv: {' '.join(sys.argv[1:])} =====\n")
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    print(f"📝 Logging console output to {ckpt_dir / 'train.log'}")


def main() -> None:
    config = get_configuration()
    _tee_console_to_logfile(config)
    setup_deterministic_training(config)
    print_configuration(config)

    worker_fn = worker_init_fn if config.system.deterministic else None
    train_loader, test_loaders, num_classes, backbone, _ = create_data_loaders(config, worker_init_fn=worker_fn)
    model, criterion, device = create_model_and_losses(config, num_classes, backbone)
    warmup_optimizer, warmup_scheduler = create_warmup_optimizer_and_scheduler(config, model)

    trainer = Trainer(config)
    trainer.setup_training(model, criterion, device, warmup_optimizer, warmup_scheduler)
    trainer.run_training_loop(train_loader, test_loaders)


if __name__ == "__main__":
    main()
