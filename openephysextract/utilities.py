from tqdm.notebook import tqdm
import psutil
import dill as pickle
import os


def savify(obj, destination, name):
    """
    Save a Python object to a .pkl file using dill.

    Parameters
    ----------
    obj : Any
        The object to be serialized.
    destination : str
        Directory to save the file.
    name : str
        Filename (with or without .pkl extension).
    """
    if not name.endswith('.pkl'):
        name += '.pkl'
    path = os.path.join(destination, name)
    os.makedirs(destination, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(obj, f)

    print(f"Saved to {path}")


def loadify(location, name):
    """
    Load a Python object from a .pkl file using dill.

    Parameters
    ----------
    location : str
        Directory containing the file.
    name : str
        Filename (with or without .pkl extension).

    Returns
    -------
    Any
        The deserialized Python object.
    """
    if not name.endswith('.pkl'):
        name += '.pkl'
    path = os.path.join(location, name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'rb') as f:
        obj = pickle.load(f)

    print(f"Loaded from {path}")
    return obj


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
