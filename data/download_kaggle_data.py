import shutil
from pathlib import Path

import kagglehub

cache_path = kagglehub.dataset_download("co1d7era/mixedtype-wafer-defect-datasets")

dest = Path(__file__).resolve().parent / "mixedtype-wafer-defect-datasets"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(cache_path, dest)
