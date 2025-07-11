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

    # statistics for optional inspection
    stats = {
        'n_channels_before': int(data.shape[0]),
        'n_channels_after': int(cleaned.shape[0]),
        'mean_distance': float(hybrid_mean.mean()),
        'std_distance': float(hybrid_mean.std())
    }

    session.stats['remove_bad'] = stats

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

    stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'max': float(np.max(data)),
        'min': float(np.min(data))
    }
    session.stats['filter_session'] = stats
    return data


def downsample_session(session: Session,
                       target_fs: int = 300) -> np.ndarray:
    """Downsample and record QC artifacts robustly."""
    data = session.preprocessed if session.preprocessed is not None else session.raw
    factor = max(1, int(session.sampling_rate // target_fs))
    downsampled = data if factor == 1 else decimate(data, factor, axis=1, ftype='fir', zero_phase=True)

    stats = {
        'mean': float(np.mean(downsampled)),
        'std': float(np.std(downsampled))
    }
    session.stats['downsample_session'] = stats
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

    stats = {
        'n_trials': int(tensor.shape[0]),
        'trial_length': span
    }
    session.stats['event_compiler'] = stats
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

    stats = {
        'n_epochs': int(epochs.shape[0]),
        'frame': frame,
        'stride': stride
    }
    session.stats['epoch_sessions'] = stats
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
        cleaned, keep = remove_bad(session, self.std, self.alpha, self.beta, self.cutoff)
        session.preprocessed = cleaned
        session.good_channels = keep
        return session

class PlotAndTraceRemoveBadStep(RemoveBadStep):
    def __init__(self, *args,
                 plot_dist: bool = True,
                 plot_traces: bool = True,
                 trace_window_sec: float = 1.0,
                 figsize=(21, 7),
                 **kwargs):
        """Same as :class:`RemoveBadStep` but displays diagnostic plots."""
        super().__init__(*args, **kwargs)
        self.plot_dist = plot_dist
        self.plot_traces = plot_traces
        self.trace_window_sec = trace_window_sec
        self.figsize = figsize

    def apply(self, session: Session) -> Session:
        data = session.raw if session.preprocessed is None else session.preprocessed
        cleaned, keep = remove_bad(session, self.std, self.alpha, self.beta, self.cutoff)
        session.preprocessed = cleaned
        session.good_channels = keep

        # recompute distances for plotting
        temp = zscore(data, axis=1) if self.std else data
        eucl = squareform(pdist(temp, metric="euclidean"))
        if eucl.max() != eucl.min():
            eucl = (eucl - eucl.min()) / (eucl.max() - eucl.min())
        corr = np.corrcoef(temp)
        corr_dist = 1 - np.abs(corr)
        hybrid = self.alpha * eucl + self.beta * corr_dist
        hybrid_mean = hybrid.mean(axis=1)
        cutoff = np.percentile(hybrid_mean, self.cutoff)
        all_idx = np.arange(data.shape[0])
        keep_mask = np.isin(all_idx, keep)

        if self.plot_dist or self.plot_traces:
            plt.figure(figsize=self.figsize)
            if self.plot_dist:
                ax1 = plt.subplot2grid((3,1), (0,0), rowspan=1)
                ax1.plot(all_idx, hybrid_mean, 'o-', alpha=0.7, label='distance')
                ax1.axhline(cutoff, color='k', ls='--', label=f'{self.cutoff}th pct')
                ax1.scatter(all_idx[~keep_mask], hybrid_mean[~keep_mask], c='red', s=50, label='removed')
                ax1.scatter(all_idx[keep_mask], hybrid_mean[keep_mask], c='darkcyan', s=50, label='kept')
                ax1.set_ylabel('Hybrid distance')
                ax1.legend(ncol=2)
            if self.plot_traces:
                n_samps = data.shape[1]
                t = np.arange(n_samps) / session.sampling_rate
                win_mask = t < self.trace_window_sec
                t_win = t[win_mask]
                ax2 = plt.subplot2grid((3,1), (1,0), rowspan=2)
                ref = data[keep[0], win_mask] if keep else data[0, win_mask]
                offset = np.ptp(ref) * 1.2 or 1.0
                for ch in all_idx:
                    trace = data[ch, win_mask]
                    centered = trace - np.mean(trace)
                    peak = np.max(np.abs(centered))
                    normed = centered / peak * offset if peak >= 1e-12 else np.zeros_like(centered)
                    y = normed + ch * offset
                    color = 'darkcyan' if keep_mask[ch] else 'red'
                    ax2.plot(t_win, y, lw=2, color=color)
                ax2.set_yticks(all_idx * offset)
                ax2.set_yticklabels([f"Ch {i}" for i in all_idx])
                ax2.set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()

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
    """Apply a user-defined sequence of SessionStep instances."""
    def __init__(self, experiment: str, sessions: List[Session], steps: List[SessionStep], destination: Optional[str] = None, raw_downsample_factor: int = 1):
        self.experiment = experiment
        self.sessions = sessions
        self.steps = steps
        self.destination = destination or os.getcwd()
        self.raw_downsample_factor = int(raw_downsample_factor)
        self.processed: List[Session] = []

    def preprocess(self, parallel: bool = False, n_jobs: Optional[int] = None, export: bool = False) -> List[Session]:
        for session in self.sessions:
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

        # downsample raw data after preprocessing for memory efficiency
        if self.raw_downsample_factor > 1:
            for session in self.processed:
                if session.raw is not None:
                    session.raw = decimate(session.raw, self.raw_downsample_factor, axis=1, ftype='fir', zero_phase=True)
                    session.sampling_rate = session.sampling_rate // self.raw_downsample_factor

        if export:
            try:
                path = os.path.join(self.destination, f"{self.experiment} PREPROCESSED.pkl")
                with open(path, "wb") as f:
                    pickle.dump(self.processed, f)
            except Exception as e:
                logger.error(f"Failed to export processed sessions: {e}")

        return self.processed

