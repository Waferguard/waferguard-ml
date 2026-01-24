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
