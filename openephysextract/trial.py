from functools import cached_property
import numpy as np

class Trial:
    __slots__ = (
        'trial', 'raw', 'data', 'source', 'location',
        'notes', 'group', 'sampling_rate'
    )

    def __init__(self, trial, raw=None, data=None, sampling_rate=30000,
                 source=None, location=None):
        """Initializes a Trial with identifiers, data, and metadata."""
        self.trial = trial
        self.raw = raw
        self.data = data  # shape: channels x samples
        self.source = source
        self.location = location
        self.notes = {}   # lightweight dict for annotations
        self.group = None
        self.sampling_rate = sampling_rate

    def __len__(self):
        """Number of channels."""
        return self.data.shape[0]

    def __getitem__(self, key):
        """Channel or slice access delegated to data array."""
        return self.data[key]

    def __repr__(self):
        return (
            f"Trial(#: {self.trial}, shape: {self.data.shape},"
            f" group: {self.group})"
        )
    @cached_property
    def duration(self) -> float:
        """Total time of trial in seconds."""
        return self.data.shape[1] / self.sampling_rate

    @cached_property
    def times(self) -> np.ndarray:
        """Time vector for samples, in seconds."""
        return np.arange(self.data.shape[1]) / self.sampling_rate

    def add_notes(self, notes: dict):
        """Merge user-provided notes into metadata."""
        if not isinstance(notes, dict):
            raise TypeError("Notes must be provided as a dict.")
        self.notes.update(notes)
        if 'group' in notes:
            self.group = notes['group']
