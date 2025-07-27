import os
import pickle
import json
import dill
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def _get_extension(name: str) -> str:
    """Extract file extension from filename, default to 'pkl' if missing."""
    ext = os.path.splitext(name)[1].lstrip('.')
    return ext if ext else 'pkl'

def savify(obj, name: str, destination: str):
    """
    Save object to structured destination.

    - Figures → destination/graphs/
    - Arrays, lists, dicts, tuples, tensors → destination/outputs/
    - All other objects → destination/models/

    Parameters:
        obj: any object (e.g. figure, data, model)
        name: filename with or without extension (e.g. 'session1.pkl', 'plot1')
        destination: root folder (e.g. '~/Documents/Research/...')
    """
    destination = Path(destination).expanduser()
    ext = _get_extension(name)
    if '.' not in name:
        name += f".{ext}"

    if isinstance(obj, plt.Figure):
        subfolder = destination / "graphs"
    elif isinstance(obj, (list, tuple, dict)) or hasattr(obj, 'shape'):
        subfolder = destination / "outputs"
    else:
        subfolder = destination / "models"

    full_path = subfolder / name
    ensure_dir(subfolder)

    if isinstance(obj, plt.Figure):
        obj.savefig(full_path)
        plt.close(obj)
    elif ext == "json":
        with open(full_path, 'w') as f:
            json.dump(obj, f, indent=2)
    elif ext in {"pkl", "dill"}:
        with open(full_path, 'wb') as f:
            try:
                pickle.dump(obj, f)
            except Exception:
                dill.dump(obj, f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print(f"Saved to {full_path}")

def loadify(name: str, location: str, obj_type: Optional[str] = None):
    """
    Load object from location.

    Parameters:
        name: filename with extension (e.g. 'session1.pkl')
        location: root folder (e.g. '~/Documents/Research/...')
        obj_type: one of ['graph', 'output', 'model'] or None

    Returns:
        Loaded object
    """
    location = Path(location).expanduser()
    subdir = {
        'graph': location / "graphs",
        'output': location / "outputs",
        'model': location / "models"
    }.get(obj_type, location)

    full_path = subdir / name
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    ext = _get_extension(name)
    if ext == "json":
        with open(full_path, 'r') as f:
            return json.load(f)
    elif ext in {"pkl", "dill"}:
        with open(full_path, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception:
                return dill.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def spreadsheet(location, name, id, relevant, sheet = None):
    """
    Load a spreadsheet (CSV or Excel) and filter it by trial IDs.
    :param location:
    :param name:
    :param id:
    :param relevant:
    :param sheet:
    :return: pd.DataFrame
    """
    filetype = name.split('.')[-1].lower()
    if filetype not in ['csv', 'xlsx']:
        raise ValueError("Unsupported file type. Only 'csv' and 'xlsx' are supported.")

    path = os.path.join(location, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if filetype == 'csv':
        df = pd.read_csv(path, sheet_name=sheet)

    elif filetype == 'xlsx':
        df = pd.read_excel(path, sheet_name=sheet)

    df = df[df[id].isin(relevant)]

    return df.reset_index(drop=True)

import logging
import tempfile
import subprocess
from tqdm import tqdm as std_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        pass  # suppress all logs in notebook

def start_log_terminal(logfile: str):
    terminal_cmds = [
        ["x-terminal-emulator", "-e", f"tail -f {logfile}"],
        ["xterm", "-e", f"tail -f {logfile}"],
        ["gnome-terminal", "--", "tail", "-f", logfile],
        ["konsole", "-e", "tail", "-f", logfile],
        ["mate-terminal", "-e", f"tail -f {logfile}"],
        ["open", "-a", "Terminal", logfile],
    ]
    for cmd in terminal_cmds:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue
    print("⚠️ Could not open external terminal window.")





