import urllib.request
import zipfile
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
zip_path = Path("./data/secom.zip")

# Make sure ./data exists
zip_path.parent.mkdir(parents=True, exist_ok=True)

# Download
urllib.request.urlretrieve(url, zip_path)

# Unzip into ./data/secom/
extract_dir = zip_path.parent / "secom"
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
