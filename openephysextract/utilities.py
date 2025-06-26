from tqdm.notebook import tqdm
import os
import pickle
import json
import dill
import psutil
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

class TqdmProgressBar:
    """
    Custom tqdm wrapper that shows memory usage and tracks the last processed item.
    """
    def __init__(self):
        self.last_file = None

    def run(self, iterable, label, func):
        with tqdm(
                iterable,
                desc=label,
                ncols=777,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"
        ) as progress:
            for item in progress:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                progress.set_postfix({
                    "Last": " " + str(self.last_file or '–')[:18],
                    "Memory": f" {mem:.1f}MB"
                })
                func(item)
                self.last_file = str(item)


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
