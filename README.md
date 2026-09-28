## 1. Create the Python environment

### Recommended method: Conda

Install Miniconda, Miniforge, or another Conda distribution.

Open Anaconda Prompt on Windows, or a terminal on macOS/Linux. Change to the project root and create the environment:

```bash
cd <your_local_path>

# Run the environment.yml file in Code_Qin:
conda env create -f /Users/<username>/Library/CloudStorage/Box-Box/Qin/Code_Qin/environment.yml
conda activate jrstation

# If missing extensions occurred, run the following:
conda install -c conda-forge \
    jupyterlab \
    jupyter_server \
    typing_extensions
```

## 2. IDE setup

The scripts are ordinary Python files and are not dependent on Spyder.
Collaborators may use JupyterLab/Notebook, Spyder, or a terminal, provided that they select the Conda environment created in Section 5.

### JupyterLab or Jupyter Notebook

If Jupyter is listed in `environment.yml`:

```bash
# The environment is named jrstation
conda activate jrstation
cd /Users/<username>/Library/CloudStorage/Box-Box/Qin/Code_Qin/JRStation
jupyter lab
```
To execute an existing script interactively from a notebook, use, for example:

```python
%run code/body.py
```

### Spyder

Start Spyder from the analysis environment or configure its Python interpreter to use the environment's executable. Open the /Users/<username>/Library/CloudStorage/Box-Box/Qin/Code_Qin/JRStation file as a Spyder project and confirm:

```python
import sys
from pathlib import Path

print(sys.executable)
print(Path.cwd())
```

Do not rely on variables remaining in Spyder's console from an earlier run. Restart the kernel and use **Run in a new dedicated console** when testing reproducibility.

## 3. Working directory and imports

The required working directory is (necessary to set in Spyder > Preference):

```text
/Users/<username>/Library/miniconda3/envs/jrstation/bin/python3.11
```

The main scripts and function module are arranged as follows:

- `JRStation/readin.py` contains dataset loading functions.
- `JRStation/analysis.py` contains reusable functions.
- `JRStation/body.py` imports those functions and performs data processing and analysis.
- `JRStation/plotting.py` produce figures, must be run after analysis results are all generated.
