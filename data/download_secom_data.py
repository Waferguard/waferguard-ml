import urllib.request
import zipfile
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
script_dir = Path(__file__).resolve().parent
zip_path = script_dir / "secom.zip"

# Make sure ./data exists
zip_path.parent.mkdir(parents=True, exist_ok=True)

# Download
urllib.request.urlretrieve(url, zip_path)

# Unzip into ./data/secom/
extract_dir = zip_path.parent / "secom"
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    extract_root = extract_dir.resolve()
    for member in z.infolist():
        # Prevent Zip Slip / path traversal by ensuring the target path stays within extract_root
        member_path = (extract_root / member.filename).resolve()
        try:
            member_path.relative_to(extract_root)
        except ValueError as e:
            raise RuntimeError(f"Unsafe path found in ZIP file: {member.filename}") from e
        z.extract(member, extract_root)

zip_path.unlink()
