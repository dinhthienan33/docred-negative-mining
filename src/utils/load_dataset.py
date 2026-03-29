"""
Download / resolve DocRED JSON paths when local ``data/`` files are missing.

Uses the Kaggle dataset ``wyldream/docred`` (same layout as standard DocRED splits).
Requires ``kagglehub`` when a fallback download is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KAGGLE_DOCRED_SLUG = "wyldream/docred"

# Basenames expected in the official DocRED / mirrored archives
_DOCRED_FILES = {
    "train_path": "train_annotated.json",
    "dev_path": "dev.json",
    "test_path": "test.json",
    "rel_info_path": "rel_info.json",
}


def _download_docred_root() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Local DocRED JSON files were not found and kagglehub is not installed. "
            "Place train_annotated.json, dev.json, test.json, and rel_info.json under "
            "your configured paths, or install kagglehub: pip install kagglehub"
        ) from exc

    path = kagglehub.dataset_download(KAGGLE_DOCRED_SLUG)
    root = Path(path)
    logger.info("Downloaded DocRED dataset to %s", root)
    return root


def _find_under(root: Path, filename: str) -> Path:
    matches = [p for p in root.rglob(filename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {root} after download."
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple matches for %s; using %s", filename, matches[0]
        )
    return matches[0]


def ensure_docred_data_paths(
    data_cfg: Dict[str, Any],
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Return a copy of ``data_cfg`` with paths pointing at existing JSON files.

    If any configured path is missing, downloads ``wyldream/docred`` via
    kagglehub once and fills missing entries by searching the cache tree.

    Args:
        data_cfg: The ``data`` section from the YAML config (paths may be
            relative to the current working directory).
        log: Optional logger (defaults to this module's logger).

    Returns:
        Updated data dict with absolute paths for the four path keys when
        resolved from download; existing local files are resolved to absolute
        paths as well.
    """
    log = log or logger
    out = dict(data_cfg)
    cache_root: Optional[Path] = None

    for key, basename in _DOCRED_FILES.items():
        raw = out.get(key)
        if raw and Path(raw).is_file():
            out[key] = str(Path(raw).resolve())
            continue

        if cache_root is None:
            log.info(
                "Local DocRED file missing for %s; downloading dataset (%s)...",
                key,
                KAGGLE_DOCRED_SLUG,
            )
            cache_root = _download_docred_root()

        found = _find_under(cache_root, basename)
        out[key] = str(found.resolve())
        log.info("Resolved %s -> %s", key, out[key])

    return out
