from functools import cached_property
import numpy as np


class Session:
    __slots__ = (
        'session', 'experiment', 'sampling_rate',
        'raw', 'preprocessed', 'data', 'shape',
        'location', 'notes', 'group', 'events', 'states',
        'ica_model', 'ica_sources', 'bad_ics',
        'good_channels', 'stats'
    )

    def __init__(self,
                 session,
                 experiment,
                 sampling_rate=30000,
                 raw=None,
                 preprocessed=None,
                 data=None,
                 location=None):
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

        # placeholder for indices of good channels after cleaning
        self.good_channels = None

        # per-step statistics for validation
        self.stats = {}

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

