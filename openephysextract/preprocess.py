import numpy as np
import pickle
import os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

from .progress import TqdmProgressBar


def remove_bad(data, std=True, alpha=0.5, beta=0.5, cutoff_percentile=90):
    """Remove outlier channels using a hybrid distance metric.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(channels, samples)``.
    std : bool
        Standardise channels before computing distances.
    alpha : float
        Weight for the Euclidean distance component.
    beta : float
        Weight for the correlation distance component.
    cutoff_percentile : int
        Percentile used to determine the distance threshold.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        The cleaned data and the indices of channels retained.
    """
    temp = zscore(data, axis=1) if std else data

    euclidean = squareform(pdist(temp, metric="euclidean"))
    if euclidean.max() != euclidean.min():
        euclidean = (euclidean - euclidean.min()) / (euclidean.max() - euclidean.min())

    corr = np.corrcoef(temp)
    corr_dist = 1 - np.abs(corr)

    hybrid = alpha * euclidean + beta * corr_dist
    np.fill_diagonal(hybrid, 1)
    hybrid_mean = hybrid.mean(axis=1)

    cutoff = np.percentile(hybrid_mean, cutoff_percentile)
    bad = np.where(hybrid_mean > cutoff)[0]
    good = [i for i in range(data.shape[0]) if i not in bad]

    return data[good, :], good


def event_compiler(trial, event_channel_number, pre_stimulus_ms=10, post_stimulus_ms=100):
    """Compile peri-event data from a ``Trial`` object.

    Parameters
    ----------
    trial : Trial
        ``Trial`` instance containing ``data`` and ``sampling_rate``.
    event_channel_number : int
        One-based index of the channel used for event detection.
    pre_stimulus_ms : int
        Window length before each event in milliseconds.
    post_stimulus_ms : int
        Window length after each event in milliseconds.

    Returns
    -------
    np.ndarray
        Tensor of shape ``(n_events, n_channels, samples_per_event)``.
    """
    if not isinstance(event_channel_number, int):
        raise TypeError("event_channel_number must be an integer")

    n_channels, n_samples = trial.data.shape
    if event_channel_number < 1 or event_channel_number > n_channels:
        raise ValueError("event_channel_number must be within channel range")

    event_chan = trial.data[event_channel_number - 1]
    threshold = 0.5
    above = event_chan >= threshold
    diffs = np.diff(above.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(event_chan) - 1)

    events = {i + 1: (int(s), int(e)) for i, (s, e) in enumerate(zip(starts, ends))}
    trial.events = events

    pre = int(pre_stimulus_ms * trial.sampling_rate / 1000)
    post = int(post_stimulus_ms * trial.sampling_rate / 1000)
    samples_per_event = pre + post
    event_tensor = np.zeros((len(starts), n_channels, samples_per_event))

    for i, start in enumerate(starts):
        start_actual = max(start - pre, 0)
        end_actual = start + post
        for j in range(n_channels):
            segment = trial.data[j, start_actual:end_actual]
            if len(segment) < samples_per_event:
                padded = np.zeros(samples_per_event)
                padded[:len(segment)] = segment
                segment = padded
            event_tensor[i, j, :] = segment[:samples_per_event]

    trial.mep_matrix = event_tensor
    return event_tensor


class Preprocessor:
    """Preprocess a list of :class:`Trial` objects."""

    def __init__(
        self,
        trials,
        destination=None,
        event_channel=None,
        remove_bad_channels=False,
        pre_stimulus_ms=10,
        post_stimulus_ms=100,
        alpha=0.5,
        beta=0.5,
        cutoff_percentile=90,
    ):
        self.trials = trials
        self.destination = destination if destination else os.getcwd()
        self.event_channel = event_channel
        self.remove = remove_bad_channels
        self.pre_ms = pre_stimulus_ms
        self.post_ms = post_stimulus_ms
        self.alpha = alpha
        self.beta = beta
        self.cutoff = cutoff_percentile

    def preprocess(self, export=False):
        """Run preprocessing on all trials."""

        def process(trial):
            if self.remove:
                cleaned, good = remove_bad(
                    trial.data,
                    std=True,
                    alpha=self.alpha,
                    beta=self.beta,
                    cutoff_percentile=self.cutoff,
                )
                trial.data = cleaned
                trial.good_channels = good
            if self.event_channel is not None:
                event_compiler(
                    trial,
                    self.event_channel,
                    pre_stimulus_ms=self.pre_ms,
                    post_stimulus_ms=self.post_ms,
                )

        progress = TqdmProgressBar()
        progress.run(self.trials, label="Preprocessing", func=process)

        if export:
            with open(os.path.join(self.destination, "preprocessed_trials.pkl"), "wb") as f:
                pickle.dump(self.trials, f)

        return self.trials
