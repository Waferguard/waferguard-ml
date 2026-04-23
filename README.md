# waferguard-ml
Github for Waferguard-ML. an AI-powered anomaly detection system designed to revolutionize semiconductor manufacturing quality assurance. This capstone project addresses the critical challenge of detecting microscopic defects in wafer fabrication before they cascade into significant yield loss and financial impact. 

# Set up

- Install python 3.11
- Install [poetry](https://python-poetry.org/docs/#installation)
- Set up virtual environment with `poetry install`

```bash
cd path/to/waferguard-ml
poetry env use python3.11
poetry install --no-root
```

## VS Code setup

- Install the Python extension for VS Code.
- Select the poetry virtual environment as the interpreter.
  - shift + cmd + p -> Python: Select Interpreter -> find the poetry environment
- Install the Ruff extension for VS Code for code formatting.

# Ruff usage
- To check for linting issues, run:
  ```bash
  poetry run ruff check .
  ```
- To automatically fix linting issues, run:
  ```bash
  poetry run ruff check . --fix
  ```
- To format the code using Ruff, run:
  ```bash
  poetry run ruff format .
  ```

# Download Datasets

## Kaggle

```bash
poetry run python data/download_kaggle_data.py
```

## Streamlit Cloud deployment note (large dataset)

`Wafer_Map_Datasets.npz` is not committed to Git because of size. The app now auto-downloads it at startup from Kaggle when needed.

Set these Streamlit Secrets in the app settings:

```toml
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_api_key"
```

Optional environment variables:

- `ENABLE_RUNTIME_DATASET_BOOTSTRAP=1` (default) enables first-run download.
- `WAFER_DATASET_SLUG=co1d7era/mixedtype-wafer-defect-datasets` overrides the Kaggle dataset slug.

For local development, Streamlit also reads secrets from `.streamlit/secrets.toml` in the project root.

Expected behavior:

- First cold start may take longer while the dataset downloads.
- Subsequent reruns in the same container reuse the local file.
- If credentials are missing or invalid, the app shows an actionable error message.

## SECOM

```bash
poetry run python data/download_secom_data.py
```