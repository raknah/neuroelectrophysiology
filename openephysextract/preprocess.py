import os
import pickle
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from scipy.signal import butter, filtfilt, decimate, iirnotch, detrend
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt

from .utilities import TqdmProgressBar
from .session import Session

# Configure logging for QC and IO operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pure preprocessing functions with robust retention hooks ---

def remove_bad(session: Session,
               std: bool = True,
               alpha: float = 0.5,
               beta: float = 0.5,
               cutoff_percentile: int = 90) -> Tuple[np.ndarray, List[int]]:
    """Remove outlier channels and record QC artifacts robustly."""
    data = session.raw if session.preprocessed is None else session.preprocessed
    temp = zscore(data, axis=1) if std else data
    eucl = squareform(pdist(temp, metric="euclidean"))
    if eucl.max() != eucl.min():
        eucl = (eucl - eucl.min()) / (eucl.max() - eucl.min())
    corr = np.corrcoef(temp)
    corr_dist = 1 - np.abs(corr)
    hybrid = alpha * eucl + beta * corr_dist
    np.fill_diagonal(hybrid, 1)
    hybrid_mean = hybrid.mean(axis=1)
    cutoff = np.percentile(hybrid_mean, cutoff_percentile)
    bad = np.where(hybrid_mean > cutoff)[0]
    keep = np.setdiff1d(np.arange(data.shape[0]), bad)
    cleaned = data[keep, :]

    # Retention: save full-resolution array
    try:
        session.save_full('remove_bad', cleaned)
    except Exception as e:
        logger.error(f"Failed to save full data for remove_bad: {e}")

    # Compute summary at session-specific rate
    rate = getattr(session, 'summary_rate', 1)
    factor = max(1, int(session.sampling_rate / rate))
    down = cleaned if factor == 1 else decimate(cleaned, factor, axis=-1)
    stats = {
        'n_channels_before': int(data.shape[0]),
        'n_channels_after': int(cleaned.shape[0]),
        'mean_distance': float(hybrid_mean.mean()),
        'std_distance': float(hybrid_mean.std())
    }
    try:
        session.store_summary('remove_bad', down, stats)
    except Exception as e:
        logger.error(f"Failed to store summary for remove_bad: {e}")

    return cleaned, list(keep)


def rereference(
        data: np.ndarray,
        reference: str = "average",
        bipolar_pairs: Optional[List[Tuple[int, int]]] = None,
        neighbors: Optional[dict] = None
) -> np.ndarray:
    """
    Re-reference multichannel data robustly.
    """
    if reference == "average":
        ref_signal = np.mean(data, axis=0)
        return data - ref_signal

    if reference == "bipolar":
        if bipolar_pairs is None:
            raise ValueError("Bipolar reference requires bipolar_pairs")
        return np.array([data[ch1] - data[ch2] for ch1, ch2 in bipolar_pairs])

    if reference == "laplacian":
        if neighbors is None:
            neighbors = {i: [] for i in range(data.shape[0])}
        reref = np.zeros_like(data)
        for i, neigh in neighbors.items():
            reref[i] = data[i] if not neigh else data[i] - np.mean(data[neigh], axis=0)
        return reref

    raise ValueError(f"Unknown reference type: {reference}")


def filter_session(session: Session,
                   lowcut: Optional[float] = None,
                   highcut: Optional[float] = None,
                   order: int = 3,
                   notch_freqs: Optional[List[float]] = None,
                   hp_cutoff: Optional[float] = None,
                   detrend_data: bool = False) -> np.ndarray:
    """Filter and record QC artifacts robustly."""
    data = session.preprocessed if session.preprocessed is not None else session.raw
    fs = session.sampling_rate
    nyq = fs * 0.5

    # Detrend
    if detrend_data:
        data = detrend(data, axis=1)
    # Notch filter
    if notch_freqs:
        for f0 in notch_freqs:
            b, a = iirnotch(f0, Q=30.0, fs=fs)
            data = filtfilt(b, a, data, axis=1)
    # High-pass filter
    if hp_cutoff:
        b, a = butter(order, hp_cutoff / nyq, btype='high')
        data = filtfilt(b, a, data, axis=1)
    # Band-pass filter
    if lowcut and highcut:
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        data = filtfilt(b, a, data, axis=1)

    # Retention
    try:
        session.save_full('filter_session', data)
    except Exception as e:
        logger.error(f"Failed to save full data for filter_session: {e}")

    rate = getattr(session, 'summary_rate', 1)
    factor = max(1, int(session.sampling_rate / rate))
    down = data if factor == 1 else decimate(data, factor, axis=-1)
    stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'max': float(np.max(data)),
        'min': float(np.min(data))
    }
    try:
        session.store_summary('filter_session', down, stats)
    except Exception as e:
        logger.error(f"Failed to store summary for filter_session: {e}")

    return data


def downsample_session(session: Session,
                       target_fs: int = 300) -> np.ndarray:
    """Downsample and record QC artifacts robustly."""
    data = session.preprocessed if session.preprocessed is not None else session.raw
    factor = max(1, int(session.sampling_rate // target_fs))
    downsampled = data if factor == 1 else decimate(data, factor, axis=1, ftype='fir', zero_phase=True)

    # Retention
    try:
        session.save_full('downsample_session', downsampled)
    except Exception as e:
        logger.error(f"Failed to save full data for downsample_session: {e}")

    stats = {
        'mean': float(np.mean(downsampled)),
        'std': float(np.std(downsampled))
    }
    try:
        session.store_summary('downsample_session', downsampled, stats)
    except Exception as e:
        logger.error(f"Failed to store summary for downsample_session: {e}")

    return downsampled


def apply_ica_cleaning(data: np.ndarray,
                       n_components: Optional[int] = None,
                       method: str = "fastica",
                       reject_strategy: str = "automated") -> Tuple[np.ndarray, np.ndarray, FastICA, List[int]]:
    """ICA cleaning without direct QC logging; logging handled in step."""
    if method != "fastica":
        raise NotImplementedError("Only 'fastica' is supported.")
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(data.T).T

    if reject_strategy == "automated":
        variances = np.var(sources, axis=1)
        bad_indices = list(np.where(np.abs(zscore(variances)) > 3)[0])
    else:
        bad_indices = []

    sources[bad_indices, :] = 0
    cleaned = ica.inverse_transform(sources.T).T
    return cleaned, sources, ica, bad_indices


def event_compiler(session: Session,
                   event_channel_number: int,
                   pre_stimulus_ms: int = 10,
                   post_stimulus_ms: int = 100,
                   baseline_correction: bool = False) -> np.ndarray:
    """Compile peri-event data and record QC artifacts."""
    n_ch, n_samp = session.preprocessed.shape
    if not isinstance(event_channel_number, int):
        raise TypeError("event_channel_number must be int")
    ev = session.preprocessed[event_channel_number - 1]
    diffs = np.diff((ev >= 0.5).astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]
    if (ev >= 0.5)[0]:
        starts = np.insert(starts, 0, 0)
    if (ev >= 0.5)[-1]:
        ends = np.append(ends, n_samp - 1)
    pre = int(pre_stimulus_ms * session.sampling_rate / 1000)
    post = int(post_stimulus_ms * session.sampling_rate / 1000)
    span = pre + post
    tensor = np.zeros((len(starts), n_ch, span))
    for i, s in enumerate(starts):
        segment = session.preprocessed[:, max(0, s-pre):s+post]
        if segment.shape[1] < span:
            pad = np.zeros((n_ch, span))
            pad[:, :segment.shape[1]] = segment
            segment = pad
        tensor[i] = segment

    # Retention
    try:
        session.save_full('peri_event', tensor)
    except Exception as e:
        logger.error(f"Failed to save full data for peri_event: {e}")
    stats = {
        'n_trials': int(tensor.shape[0]),
        'trial_length': span
    }
    summary = tensor.mean(axis=(0,2)) if tensor.size else np.array([])
    try:
        session.store_summary('peri_event', summary, stats)
    except Exception as e:
        logger.error(f"Failed to store summary for peri_event: {e}")
    return tensor


def epoch_sessions(session: Session,
                   frame: int,
                   stride: int,
                   autoreject: bool = True,
                   threshold: Optional[float] = None,
                   consensus: float = 0.6) -> np.ndarray:
    """Epoch data and record QC artifacts."""
    ch, samp = session.preprocessed.shape
    data_w = np.lib.stride_tricks.sliding_window_view(session.preprocessed, (frame,), axis=1)
    windows = data_w[:, ::stride, :]
    epochs = np.moveaxis(windows, 0, 1)

    if autoreject:
        ptps = np.ptp(epochs, axis=2).flatten()
        if threshold is None:
            med, mad = np.median(ptps), np.median(np.abs(ptps - np.median(ptps)))
            threshold = med + 5 * mad
        clean_epochs = [ep for ep in epochs if np.sum(np.ptp(ep, axis=1) > threshold)/ch < (1 - consensus)]
        epochs = np.stack(clean_epochs, axis=0) if clean_epochs else np.empty((0, ch, frame))

    # Retention
    try:
        session.save_full('epochs', epochs)
    except Exception as e:
        logger.error(f"Failed to save full data for epochs: {e}")
    stats = {
        'n_epochs': int(epochs.shape[0]),
        'frame': frame,
        'stride': stride
    }
    summary = epochs.mean(axis=(0,2)) if epochs.size else np.array([])
    try:
        session.store_summary('epochs', summary, stats)
    except Exception as e:
        logger.error(f"Failed to store summary for epochs: {e}")
    return epochs

# --- Step abstraction and implementations ---
# --- Step abstraction and implementations ---

class SessionStep(ABC):
    """A preprocessing step for a single Session."""
    @abstractmethod
    def apply(self, session: Session) -> Session:
        pass


class RemoveBadStep(SessionStep):
    def __init__(self, std=True, alpha=0.5, beta=0.5, cutoff_percentile=90):
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.cutoff = cutoff_percentile

    def apply(self, session: Session) -> Session:
        cleaned, _ = remove_bad(session, self.std, self.alpha, self.beta, self.cutoff)
        session.preprocessed = cleaned
        return session


class ReReferenceStep(SessionStep):
    def __init__(self, reference="average", bipolar_pairs=None):
        self.reference = reference
        self.bipolar_pairs = bipolar_pairs

    def apply(self, session: Session) -> Session:
        data = session.preprocessed if session.preprocessed is not None else session.raw
        reref_data = rereference(data, self.reference, self.bipolar_pairs)
        session.preprocessed = reref_data
        return session


class FilterStep(SessionStep):
    def __init__(self, lowcut=None, highcut=None, order=3, notch_freqs=None, hp_cutoff=None, detrend_data=False):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freqs = notch_freqs
        self.hp_cutoff = hp_cutoff
        self.detrend_data = detrend_data

    def apply(self, session: Session) -> Session:
        filtered = filter_session(session, self.lowcut, self.highcut, self.order,
                                  self.notch_freqs, self.hp_cutoff, self.detrend_data)
        session.preprocessed = filtered
        return session


class DownsampleStep(SessionStep):
    def __init__(self, target_fs: int = 300):
        self.target_fs = target_fs

    def apply(self, session: Session) -> Session:
        downsampled = downsample_session(session, self.target_fs)
        session.preprocessed = downsampled
        session.sampling_rate = self.target_fs
        return session


class ICARemovalStep(SessionStep):
    def __init__(self, n_components=None, method="fastica", reject_strategy="automated"):
        self.n_components = n_components
        self.method = method
        self.reject_strategy = reject_strategy

    def apply(self, session: Session) -> Session:
        data = session.preprocessed if session.preprocessed is not None else session.raw
        cleaned, sources, ica_model, bad_indices = apply_ica_cleaning(
            data, self.n_components, self.method, self.reject_strategy
        )
        session.preprocessed = cleaned
        session.ica_sources = sources
        session.ica_model = ica_model
        session.bad_ics = bad_indices
        return session


class EventCompileStep(SessionStep):
    def __init__(self, event_channel: int, pre_ms=10, post_ms=100, baseline_correction=False):
        self.event_channel = event_channel
        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.baseline_correction = baseline_correction

    def apply(self, session: Session) -> Session:
        session.data = event_compiler(session, self.event_channel, self.pre_ms, self.post_ms, self.baseline_correction)
        return session


class EpochStep(SessionStep):
    def __init__(self, frame: int, stride: int, autoreject=True, threshold=None, consensus=0.6):
        self.frame = frame
        self.stride = stride
        self.autoreject = autoreject
        self.threshold = threshold
        self.consensus = consensus

    def apply(self, session: Session) -> Session:
        session.data = epoch_sessions(session, self.frame, self.stride, self.autoreject, self.threshold, self.consensus)
        return session


# --- Core Preprocessor ---

class Preprocessor:
    """Apply a user-defined sequence of SessionStep instances with robust QC support."""
    def __init__(self, experiment: str, sessions: List[Session], steps: List[SessionStep], destination: Optional[str] = None, summary_rate: int = 1):
        self.experiment = experiment
        self.sessions = sessions
        self.steps = steps
        self.destination = destination or os.getcwd()
        self.summary_rate = summary_rate
        self.processed: List[Session] = []

    def preprocess(self, parallel: bool = False, n_jobs: Optional[int] = None, export: bool = False) -> List[Session]:
        for session in self.sessions:
            setattr(session, 'summary_rate', self.summary_rate)
            if session.raw is None and session.data is not None:
                session.raw = session.data
            session.preprocessed = None
            session.data = None

        def _apply(session: Session) -> Session:
            try:
                for step in self.steps:
                    session = step.apply(session)
            except Exception as e:
                logger.error(f"Error in preprocessing session {session.session}: {e}")
                raise
            return session

        if parallel:
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                self.processed = list(exe.map(_apply, self.sessions))
        else:
            progress = TqdmProgressBar()
            self.processed = []
            def wrapped_apply(sess):
                res = _apply(sess)
                self.processed.append(res)
            progress.run(self.sessions, label="Preprocessing", func=wrapped_apply)

        if export:
            try:
                path = os.path.join(self.destination, f"{self.experiment} PREPROCESSED.pkl")
                with open(path, "wb") as f:
                    pickle.dump(self.processed, f)
            except Exception as e:
                logger.error(f"Failed to export processed sessions: {e}")

        return self.processed
