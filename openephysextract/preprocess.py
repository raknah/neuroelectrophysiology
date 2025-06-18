import os
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from scipy.signal import butter, filtfilt, decimate

from .utilities import TqdmProgressBar
from .trial import Trial

# --- Pure preprocessing functions ---

def remove_bad(data: np.ndarray,
               std: bool = True,
               alpha: float = 0.5,
               beta: float = 0.5,
               cutoff_percentile: int = 90) -> tuple[np.ndarray, List[int]]:
    """Remove outlier channels using a hybrid distance metric."""
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
    good = [i for i in range(data.shape[0]) if i not in bad]
    return data[good, :], good


def event_compiler(trial: Trial,
                   event_channel_number: int,
                   pre_stimulus_ms: int = 10,
                   post_stimulus_ms: int = 100) -> np.ndarray:
    """Compile peri-event data from a Trial object."""
    n_ch, n_samp = trial.data.shape
    if not isinstance(event_channel_number, int):
        raise TypeError("event_channel_number must be an integer")
    if not (1 <= event_channel_number <= n_ch):
        raise ValueError("event_channel_number out of range")
    ev = trial.data[event_channel_number - 1]
    thresh = 0.5
    above = ev >= thresh
    diffs = np.diff(above.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, n_samp - 1)
    trial.events = {i + 1: (int(s), int(e)) for i, (s, e) in enumerate(zip(starts, ends))}
    pre = int(pre_stimulus_ms * trial.sampling_rate / 1000)
    post = int(post_stimulus_ms * trial.sampling_rate / 1000)
    span = pre + post
    tensor = np.zeros((len(starts), n_ch, span))
    for i, s in enumerate(starts):
        start = max(s - pre, 0)
        end = s + post
        seg = trial.data[:, start:end]
        if seg.shape[1] < span:
            pad = np.zeros((n_ch, span))
            pad[:, :seg.shape[1]] = seg
            seg = pad
        tensor[i, :, :] = seg[:, :span]
    trial.mep_matrix = tensor
    return tensor


def epoch(trials: List[Trial], frame: int, stride: int) -> List[Trial]:
    """Split each Trial into overlapping windows."""
    epoched: List[Trial] = []
    for trial in trials:
        ch, samp = trial.data.shape
        n = ((samp - frame) // stride) + 1
        data = np.lib.stride_tricks.sliding_window_view(trial.data, window_shape=(frame,), axis=1)
        windows = data[:, ::stride, :][:, :n, :]
        windows = np.moveaxis(windows, 0, 1)
        epoched.append(Trial(
            trial=trial.trial,
            raw=trial.data,
            data=windows,
            sampling_rate=trial.sampling_rate,
            source=trial.notes.get('source'),
            location=trial.notes.get('location')
        ))
    return epoched


def filter(trials: List[Trial],
           lowcut: float,
           highcut: float,
           order: int = 3) -> List[Trial]:
    """Apply bandpass filter to all channels in each Trial."""

    out: List[Trial] = []
    for trial in trials:
        fs = trial.sampling_rate
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        fd = filtfilt(b, a, trial.data, axis=1)
        out.append(Trial(
            trial=trial.trial,
            data=fd,
            sampling_rate=fs,
            source=trial.notes.get('source'),
            location=trial.notes.get('location')
        ))
    return out


def downsample(trial: Trial, target_fs: int = 300) -> Trial:
    """Downsample a single Trial to target_fs using FIR decimation."""
    factor = int(trial.sampling_rate // target_fs)
    ds = decimate(trial.data, factor, axis=1, ftype='fir', zero_phase=True)
    return Trial(
        trial=trial.trial,
        data=ds,
        sampling_rate=target_fs,
        source=trial.notes.get('source'),
        location=trial.notes.get('location')
    )

# --- Step abstraction and implementations ---
class TrialStep(ABC):
    """A preprocessing step for a single Trial."""
    @abstractmethod
    def apply(self, trial: Trial) -> Trial:
        pass

class RemoveBadStep(TrialStep):
    def __init__(self, std: bool = True, alpha: float = 0.5, beta: float = 0.5, cutoff_percentile: int = 90):
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.cutoff = cutoff_percentile

    def apply(self, trial: Trial) -> Trial:
        cleaned, _ = remove_bad(
            trial.data,
            std=self.std,
            alpha=self.alpha,
            beta=self.beta,
            cutoff_percentile=self.cutoff
        )
        trial.data = cleaned
        return trial

class EventCompileStep(TrialStep):
    def __init__(self, event_channel: int, pre_ms: int = 10, post_ms: int = 100):
        self.event_channel = event_channel
        self.pre_ms = pre_ms
        self.post_ms = post_ms

    def apply(self, trial: Trial) -> Trial:
        event_compiler(
            trial,
            self.event_channel,
            pre_stimulus_ms=self.pre_ms,
            post_stimulus_ms=self.post_ms
        )
        return trial

class EpochStep(TrialStep):
    def __init__(self, frame: int, stride: int):
        self.frame = frame
        self.stride = stride

    def apply(self, trial: Trial) -> Trial:
        return epoch([trial], frame=self.frame, stride=self.stride)[0]

class FilterStep(TrialStep):
    def __init__(self, lowcut: float, highcut: float, order: int = 3):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def apply(self, trial: Trial) -> Trial:
        return filter([trial], self.lowcut, self.highcut, self.order)[0]

class DownsampleStep(TrialStep):
    def __init__(self, target_fs: int = 300):
        self.target_fs = target_fs

    def apply(self, trial: Trial) -> Trial:
        return downsample(trial, self.target_fs)

# --- Core Preprocessor ---
class Preprocessor:
    """Apply a user-defined sequence of TrialStep instances to each Trial."""
    def __init__(
            self,
            trials: List[Trial],
            steps: List[TrialStep],
            destination: Optional[str] = None
    ):
        self.trials = trials
        self.steps = steps
        self.destination = destination or os.getcwd()
        self.processed: List[Trial] = []

    def preprocess(
            self,
            parallel: bool = False,
            n_jobs: Optional[int] = None,
            export: bool = False
    ) -> List[Trial]:
        """Run steps on all trials, optionally in parallel, and optionally export."""
        def _apply(trial: Trial) -> Trial:
            for step in self.steps:
                trial = step.apply(trial)
            return trial

        if parallel:
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                self.processed = list(exe.map(_apply, self.trials))
        else:
            progress = TqdmProgressBar()
            self.processed = []
            def wrapped_apply(trial):
                result = _apply(trial)
                self.processed.append(result)
            progress.run(self.trials, label="Preprocessing", func=wrapped_apply)

        if export:
            path = os.path.join(self.destination, "preprocessed_trials.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.processed, f)

        return self.processed
