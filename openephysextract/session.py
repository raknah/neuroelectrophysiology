from functools import cached_property
import numpy as np
from pathlib import Path

class Session:
    __slots__ = (
        'session', 'experiment', 'sampling_rate',
        'raw', 'preprocessed', 'data', 'shape',
        'location', 'notes', 'group', 'events', 'states',
        'ica_model', 'ica_sources', 'bad_ics',
        # New slots for QC
        'output_dir', 'data_pointers', 'qc_summaries', 'summary_rate'
    )

    def __init__(self,
                 session,
                 experiment,
                 sampling_rate=30000,
                 raw=None,
                 preprocessed=None,
                 data=None,
                 location=None,
                 output_dir: str = None):
        """Initializes a session with identifiers, data, metadata, and QC storage."""
        # existing initialization
        self.session = session
        self.experiment = experiment
        self.raw = raw
        self.preprocessed = preprocessed
        self.data = data
        self.shape = (self.data.shape if self.data is not None else
                      self.raw.shape if self.raw is not None else (0, 0))
        self.location = location
        self.notes = {}
        self.group = None
        self.sampling_rate = sampling_rate
        self.events = None
        self.states = None
        self.ica_model = None
        self.ica_sources = None
        self.bad_ics = None

        # QC-specific initialization
        # Determine where to save full-resolution arrays
        base = Path(output_dir) if output_dir else Path.cwd() / 'qc_output'
        self.output_dir = base / f"{self.experiment}_{self.session}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # pointers to disk-stored full arrays
        self.data_pointers = {}
        # lightweight summaries in RAM for QC
        self.qc_summaries = {}
        self.summary_rate = None

    # retain existing dunder methods
    def __len__(self):
        return self.data.shape[-1] if self.data is not None else self.raw.shape[-1]

    def __getitem__(self, key):
        return self.data[key] if self.data is not None else self.raw[key]

    def __repr__(self):
        return (
            f"session(#: {self.session}, shape: {self.shape}, group: {self.group})"
        )

    @cached_property
    def duration(self) -> float:
        return self.data.shape[1] / self.sampling_rate

    @cached_property
    def times(self) -> np.ndarray:
        return np.arange(self.data.shape[1]) / self.sampling_rate

    def add_notes(self, notes: dict):
        if not isinstance(notes, dict):
            raise TypeError("Notes must be provided as a dict.")
        self.notes.update(notes)
        if 'group' in notes:
            self.group = notes['group']
        else:
            raise ValueError("Notes must contain a 'group' key to set the group or phenotype.")

    # --- QC retention policy methods ---
    def save_full(self, name: str, array: np.ndarray):
        """Save full-resolution array to disk and record pointer."""
        file_path = self.output_dir / f"{name}.npy"
        np.save(file_path, array)
        self.data_pointers[name] = {
            'path': file_path,
            'shape': array.shape,
            'dtype': array.dtype.name
        }

    def load_full(self, name: str) -> np.ndarray:
        """Memory-map and load a disk-stored array."""
        info = self.data_pointers.get(name)
        if info is None:
            raise KeyError(f"No data pointer for '{name}'")
        return np.load(info['path'], mmap_mode='r')

    def store_summary(self, name: str, downsampled: np.ndarray, stats: dict):
        """Keep lightweight summary in RAM for QC plotting."""
        self.qc_summaries[name] = {
            'downsampled': downsampled,
            'stats': stats
        }