"""Utilities for downloading and staging the wafer dataset from Kaggle."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_DATASET = "co1d7era/mixedtype-wafer-defect-datasets"
DEFAULT_DEST_DIR = Path(__file__).resolve().parent / "mixedtype-wafer-defect-datasets"
DEFAULT_DEST_FILE = DEFAULT_DEST_DIR / "Wafer_Map_Datasets.npz"


def find_dataset_file(root_dir: Path, filename: str = "Wafer_Map_Datasets.npz") -> Path | None:
    """Find the first matching dataset file beneath a directory tree."""
    matches = sorted(root_dir.rglob(filename))
    if matches:
        logger.info("Found %s candidate(s) for %s under %s", len(matches), filename, root_dir)
        return matches[0]

    logger.warning("No %s file found under %s", filename, root_dir)
    return None


def configure_kaggle_credentials(username: str | None = None, key: str | None = None) -> None:
    """Configure Kaggle credentials via env vars and kaggle.json for cloud runtime use."""
    if not username or not key:
        logger.info("Kaggle credentials not provided; relying on existing environment or secrets state.")
        return

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    kaggle_json.chmod(0o600)
    logger.info("Kaggle credentials configured at %s", kaggle_json)


def copy_dataset_tree(dataset_slug: str, dest_dir: Path) -> Path:
    """Download dataset slug via kagglehub and copy tree to destination directory."""
    logger.info("Downloading Kaggle dataset %s", dataset_slug)
    cache_path = Path(kagglehub.dataset_download(dataset_slug))
    logger.info("Kaggle dataset cached at %s", cache_path)
    logger.info("Cached dataset entries: %s", [p.name for p in sorted(cache_path.iterdir())[:10]])
    if dest_dir.exists():
        logger.info("Removing existing dataset directory %s", dest_dir)
        shutil.rmtree(dest_dir)
    shutil.copytree(cache_path, dest_dir)
    logger.info("Copied dataset tree to %s", dest_dir)
    return dest_dir


def ensure_wafer_dataset(
    dataset_file: Path,
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    username: str | None = None,
    key: str | None = None,
    force: bool = False,
) -> Path:
    """Ensure the wafer dataset file exists locally; download if missing or forced."""
    dataset_file = dataset_file.resolve()
    if dataset_file.exists() and not force:
        logger.info("Dataset already present at %s", dataset_file)
        return dataset_file

    logger.info(
        "Ensuring wafer dataset at %s (slug=%s, force=%s)",
        dataset_file,
        dataset_slug,
        force,
    )
    configure_kaggle_credentials(username=username, key=key)
    dest_dir = dataset_file.parent
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        copy_dataset_tree(dataset_slug=dataset_slug, dest_dir=dest_dir)
    except Exception:
        logger.exception("Kaggle dataset download failed for slug %s", dataset_slug)
        raise

    if not dataset_file.exists():
        logger.warning("Expected dataset file missing after copy: %s", dataset_file)
        candidate = find_dataset_file(dest_dir)
        if candidate is not None:
            logger.info("Copying discovered dataset file %s to expected path %s", candidate, dataset_file)
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dataset_file)

    if not dataset_file.exists():
        msg = f"Dataset download completed but expected file not found: {dataset_file}"
        logger.error(
            "Dataset directory contents under %s: %s", dest_dir, [p.name for p in sorted(dest_dir.iterdir())[:20]]
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Dataset ready at %s (%s bytes)", dataset_file, dataset_file.stat().st_size)
    return dataset_file


def parse_args() -> argparse.Namespace:
    """Parse command line options for local dataset download workflow."""
    parser = argparse.ArgumentParser(description="Download wafer map dataset from Kaggle")
    parser.add_argument("--dataset-slug", default=DEFAULT_KAGGLE_DATASET, help="Kaggle dataset slug")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for local usage."""
    args = parse_args()
    dataset_file = ensure_wafer_dataset(
        dataset_file=DEFAULT_DEST_FILE,
        dataset_slug=args.dataset_slug,
        force=args.force,
    )
    print(f"Dataset ready: {dataset_file}")


if __name__ == "__main__":
    main()
