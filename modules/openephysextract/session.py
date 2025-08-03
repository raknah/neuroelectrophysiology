import os
import json
import numpy as np
import h5py

from datetime import datetime
from functools import cached_property
from typing import List, Optional, Sequence, Tuple, Dict, Any


class Session:
    """
    Represents an EEG/MEG recording session with raw data, preprocessing,
    metadata, and convenient serialization to/from HDF5 for cross-language analysis.
    """

    __slots__ = (
        'session', 'experiment', 'sampling_rate',
        'raw', 'preprocessed', 'data',
        'location', 'ch_names', 'montage',
        'notes', 'group', 'events', 'stats', 'history',
        'states', 'ica_model', 'ica_sources', 'bad_ics',
        'good_channels', 'original_channels', 'schema_version'
    )

    def __init__(
        self,
        session: str,
        experiment: str,
        raw: np.ndarray,
        preprocessed: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        sampling_rate: int = 30000,
        ch_names: Optional[List[str]] = None,
        montage: Optional[np.ndarray] = None,
        location: Optional[str] = None,
        original_channels: Optional[List[int]] = None,
    ):
        """
        Initialize a Session.

        Args:
            session: Unique session identifier.
            experiment: Parent experiment name or ID.
            raw: Raw data array (n_channels, n_samples), NumPy array.
            preprocessed: Optionally pre-filtered data.
            data: Final processed output (e.g. epoched).
            sampling_rate: Sampling rate in Hz.
            ch_names: Optional channel labels.
            montage: Optional electrode positions as (n_channels, 3) array.
            location: Recording location descriptor.
            original_channels: Original channel indices from hardware (e.g., [3,4,5,6,7,8]).
        """
        # Metadata
        self.session = session
        self.experiment = experiment
        self.schema_version = 1

        # Data arrays (NumPy)
        self.raw = np.ascontiguousarray(raw, dtype=np.float32)
        self.preprocessed = (
            np.ascontiguousarray(preprocessed, dtype=np.float32)
            if preprocessed is not None
            else None
        )
        self.data = (
            np.ascontiguousarray(data, dtype=np.float32)
            if data is not None
            else None
        )

        # Recording params
        self.sampling_rate = sampling_rate
        self.ch_names = ch_names or []
        self.montage = montage
        self.location = location
        self.original_channels = original_channels

        # Provenance & analysis metadata
        self.notes: Dict[str, Any] = {'schema_version': self.schema_version}
        self.group: Optional[str] = None
        self.events: Optional[List[Tuple[int, str]]] = None
        self.stats: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

        # Optional downstream results
        self.states: Any = None
        self.ica_model: Any = None
        self.ica_sources: Optional[np.ndarray] = None
        self.bad_ics: Optional[List[int]] = None
        self.good_channels: Optional[List[int]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.data is not None:
            arr = self.data
        elif self.preprocessed is not None:
            arr = self.preprocessed
        elif self.raw is not None:
            arr = self.raw
        else:
            raise ValueError("No data found in session")
        return tuple(arr.shape)
    def __len__(self) -> int:
        """Number of timepoints in the active data array."""
        return self.shape[1]

    def __getitem__(self, key) -> np.ndarray:
        """Index into the active array (data → preprocessed → raw)."""
        if self.data is not None:
            arr = self.data
        elif self.preprocessed is not None:
            arr = self.preprocessed
        else:
            arr = self.raw
        return arr[key]

    def __repr__(self) -> str:
        return f"Session('{self.session}', shape={self.shape}, group={self.group})"

    @cached_property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return self.shape[1] / self.sampling_rate

    @cached_property
    def times(self) -> np.ndarray:
        """Time axis vector (in seconds) aligned to samples."""
        return np.arange(self.shape[1], dtype=np.float32) / self.sampling_rate

    def add_notes(self, notes: Dict[str, Any]) -> None:
        """Merge arbitrary metadata into session.notes."""
        if not isinstance(notes, dict):
            raise TypeError("notes must be a dict")
        self.notes.update(notes)
        if 'group' in notes:
            self.group = notes['group']

    def annotate(self, onsets: Sequence[int], labels: Sequence[str]) -> None:
        """Attach event annotations: sample indices + labels."""
        if len(onsets) != len(labels):
            raise ValueError("onsets and labels must have equal length")
        self.events = list(zip(onsets, labels))

    def log_step(self, step: Any) -> None:
        """
        Log a preprocessing or analysis step.
        Records step name, parameters, and UTC timestamp.
        """
        entry = {
            'step': step.__class__.__name__,
            'params': {k: v for k, v in vars(step).items() if not k.startswith('_')},
            'time': datetime.utcnow().isoformat()
        }
        self.history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata (non-array) to a JSON-serializable dict."""
        return {
            'session': self.session,
            'experiment': self.experiment,
            'sampling_rate': self.sampling_rate,
            'location': self.location,
            'ch_names': self.ch_names,
            'montage': self.montage.tolist() if isinstance(self.montage, np.ndarray) else self.montage,
            'notes': self.notes,
            'group': self.group,
            'events': self.events,
            'stats': self.stats,
            'history': self.history,
            'good_channels': self.good_channels,
            'original_channels': self.original_channels,
            'bad_ics': self.bad_ics
        }

    def to_hdf5(self, path: str) -> None:
        """
        Serialize the session to a single HDF5 file.
        - Datasets: raw, preprocessed, data
        - Attributes: all fields from to_dict()
        """
        print(f"Saving to {path}...")
        print(f"  - Raw data shape: {self.raw.shape if self.raw is not None else 'None'}")
        if self.preprocessed is not None:
            print(f"  - Preprocessed data shape: {self.preprocessed.shape}")
        if self.data is not None:
            print(f"  - Final data shape: {self.data.shape}")

        with h5py.File(path, 'w') as f:
            f.create_dataset('raw', data=self.raw, compression='gzip')
            if self.preprocessed is not None:
                f.create_dataset('preprocessed', data=self.preprocessed, compression='gzip')
            if self.data is not None:
                f.create_dataset('data', data=self.data, compression='gzip')

            for k, v in self.to_dict().items():
                if v is None:
                    continue
                if isinstance(v, (int, float, str, bool)):
                    f.attrs[k] = v
                else:
                    f.attrs[k] = json.dumps(v)

    @classmethod
    def from_dict(cls, meta: Dict[str, Any], raw: np.ndarray) -> 'Session':
        """Reconstruct a Session from metadata dict and raw NumPy array."""
        montage = np.array(meta['montage']) if meta.get('montage') is not None else None
        sess = cls(
            session=meta['session'],
            experiment=meta['experiment'],
            raw=raw,
            sampling_rate=meta['sampling_rate'],
            ch_names=meta.get('ch_names'),
            montage=montage,
            location=meta.get('location'),
        )
        sess.notes = meta.get('notes', {})
        sess.group = meta.get('group')
        sess.events = meta.get('events')
        sess.stats = meta.get('stats', {})
        sess.history = meta.get('history', [])
        sess.good_channels = meta.get('good_channels')
        sess.original_channels = meta.get('original_channels')
        sess.bad_ics = meta.get('bad_ics')
        return sess

    @classmethod
    def from_hdf5(cls, path: str) -> 'Session':
        """Load a session previously saved with to_hdf5()."""
        with h5py.File(path, 'r') as f:
            raw = f['raw'][()]
            pre = f['preprocessed'][()] if 'preprocessed' in f else None
            data = f['data'][()] if 'data' in f else None
            meta = {}
            for k in f.attrs:
                val = f.attrs[k]
                try:
                    decoded = val.decode() if isinstance(val, bytes) else str(val)
                    meta[k] = json.loads(decoded)
                except Exception:
                    meta[k] = val

        sess = cls.from_dict(meta, raw)
        sess.preprocessed = pre
        sess.data = data
        return sess

    def save(self, directory: str) -> None:
        """Sugar: Save to '{directory}/{session}.h5'."""
        os.makedirs(directory, exist_ok=True)
        self.to_hdf5(os.path.join(directory, f"{self.session}.h5"))

    @classmethod
    def load(cls, directory: str, session_id: str) -> 'Session':
        """Sugar: Load from '{directory}/{session_id}.h5'."""
        return cls.from_hdf5(os.path.join(directory, f"{session_id}.h5"))
