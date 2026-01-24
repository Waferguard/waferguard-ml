import shutil
from pathlib import Path

import kagglehub

cache_path = kagglehub.dataset_download("co1d7era/mixedtype-wafer-defect-datasets")

dest = Path("./data/") / "mixedtype-wafer-defect-datasets"
shutil.copytree(cache_path, dest, dirs_exist_ok=True)
