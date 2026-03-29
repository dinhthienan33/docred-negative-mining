"""
General utility functions for the DocRED pipeline.

Covers: reproducibility seeding, device detection, parameter counting,
config loading, logging setup, and metrics formatting.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect and return the best available compute device.

    Returns:
        ``torch.device("cuda")`` when an NVIDIA GPU is available,
        ``torch.device("mps")`` on Apple Silicon, or
        ``torch.device("cpu")`` as fallback.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using CUDA device: %s (device count=%d)",
            torch.cuda.get_device_name(0),
            torch.cuda.device_count(),
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected; using CPU.")
    return device


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of *trainable* parameters in a model.

    Args:
        model: Any :class:`torch.nn.Module`.

    Returns:
        Integer count of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Trainable parameters: %s (%.2fM)",
        f"{total:,}",
        total / 1e6,
    )
    return total


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", path)
    return config


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into a *base* config dict.

    Args:
        base: Base configuration dictionary.
        overrides: Dict of override values (may be nested).

    Returns:
        Merged configuration dict (the *base* dict is modified in-place and
        also returned).
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def create_logger(
    output_dir: Union[str, Path],
    log_level: int = logging.INFO,
    name: str = "docred",
) -> logging.Logger:
    """Configure and return a named logger that writes to stdout and a file.

    Creates ``output_dir`` if it does not exist.  Log lines are written to
    both the console and ``<output_dir>/train.log``.

    Args:
        output_dir: Directory for the log file.
        log_level: Logging verbosity (default: :data:`logging.INFO`).
        name: Logger name (default: ``"docred"``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_logger = logging.getLogger(name)
    log_logger.setLevel(log_level)

    if not log_logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        log_logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        log_logger.addHandler(fh)

    return log_logger


# ---------------------------------------------------------------------------
# Metrics formatting
# ---------------------------------------------------------------------------

def format_metrics(metrics_dict: Dict[str, Any], prefix: str = "") -> str:
    """Format a metrics dictionary into a human-readable string.

    Args:
        metrics_dict: Keys are metric names, values are scalars.
        prefix: Optional string prefix added before each line.

    Returns:
        Multi-line formatted string.

    Example::

        >>> format_metrics({"f1": 0.6789, "ign_f1": 0.6543, "loss": 1.234})
        '  f1       : 0.6789\\n  ign_f1   : 0.6543\\n  loss     : 1.2340'
    """
    if not metrics_dict:
        return ""

    max_key_len = max(len(k) for k in metrics_dict)
    lines = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            formatted_val = f"{value:.4f}"
        else:
            formatted_val = str(value)
        lines.append(f"{prefix}{key:<{max_key_len}} : {formatted_val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "best_model.pt",
) -> Path:
    """Save a training checkpoint to disk.

    Args:
        state: Dict containing model / optimizer state dicts, epoch number,
            and metric values.
        output_dir: Directory to write the checkpoint file.
        filename: Checkpoint file name.

    Returns:
        Absolute path to the saved checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename
    torch.save(state, save_path)
    logger.info("Checkpoint saved to %s", save_path)
    return save_path


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        device: Device to map tensors to.  If ``None``, uses CPU.

    Returns:
        Checkpoint state dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = device if device is not None else torch.device("cpu")
    state = torch.load(path, map_location=map_location)
    logger.info("Loaded checkpoint from %s", path)
    return state
