import os
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

from .progress import TqdmProgressBar


def remove_bad_channels(data, std, alpha, beta, cutoff_percentile = 90):

    """
    Removes bad channels based on a hybrid distance metric combining z-scored data,
    pairwise Euclidean distance, and correlation distance.

    Parameters:
        data (numpy.ndarray): 2D array of shape (n_channels, n_samples).
        alpha (float): Weight for the Euclidean distance component.
        beta (float): Weight for the correlation distance component.
        cutoff_percentile (int): Percentile threshold for identifying bad channels.

    Returns:
        numpy.ndarray: 2D array of z-scored data with bad channels removed.
    """

    temp = data

    if std:
        temp = zscore(temp, axis = 1)

    # pairwise distance (euclidean)
    euclidean = squareform(pdist(temp, metric = 'euclidean'))
    euclidean = (euclidean - euclidean.min()) / (euclidean.max() - euclidean.min())

    # correlation
    corr = np.corrcoef(temp)
    corr_dist = 1 - np.abs(corr)

    hybrid = alpha * euclidean + beta * corr_dist
    np.fill_diagonal(hybrid, 1)
    hybrid_mean = hybrid.mean(axis = 1)

    cutoff = np.percentile(hybrid_mean, cutoff_percentile)
    bad = np.where(hybrid_mean > cutoff)[0]

    print("Bad channel indices (relative to chs):", bad)
    print("Mean distances: \n", hybrid)
    print("Cutoff value:", cutoff)

    good = [i for i in range(6) if i not in bad]

    return temp[good, :]

def event_compiler(trial, event_channel_number: int, export=True):
    """
    Detect events in a specified channel, then extract peri‐event data across all recording channels,
    returning a tensor of shape (n_events, n_channels, n_samples_per_event).

    This method:
        1. Checks that `event_channel_number` is a valid integer within channel range.
        2. Thresholds the specified event channel at 0.5 to find event start/end indices.
        3. Stores a dict of {event_index: (start_sample, end_sample)} in `self.events`.
        4. For each detected event, extracts a window of raw data from
           (start – pre_stimulus) to (start + post_stimulus) for every channel in `self.recording_channels`.
        5. Assembles these windows into a NumPy array of shape
           (n_events, n_channels, n_samples_per_event) and assigns it to `self.mep_matrix`.
        6. Optionally exports both the event dictionary and the tensor via `self._save_object`.

    Parameters:
        event_channel_number (int):  One‐based index of the channel used for event detection.
        export (bool, optional):    If True (default), save the detected‐events dict and data tensor.
        export_format (str, optional):  File format for export; one of 'pkl', 'json', or 'mat'. Default is 'pkl'.

    Returns:
        numpy.ndarray:  A 3D array with shape (n_events, n_channels, n_samples_per_event).

    Raises:
        TypeError:  If `event_channel_number` is not an integer.
        ValueError: If `event_channel_number` is outside the range [1, self.num_channels].
    """
    # --- Validate event_channel_number ---
    if not isinstance(event_channel_number, int):
        raise TypeError("`event_channel_number` must be an integer")
    if event_channel_number < 1 or event_channel_number > self.num_channels:
        raise ValueError("`event_channel_number` must be within the valid range of channels")

    # --- Detect event locations ---
    data = self.raw_data
    threshold = 0.5
    # Convert one‐based channel index to zero‐based for slicing
    event_chan_data = data[:, event_channel_number - 1]
    above_thresh = event_chan_data >= threshold

    # Find rising edges (event starts) and falling edges (event ends)
    diffs = np.diff(above_thresh.astype(int))
    event_start_indices = np.where(diffs == 1)[0] + 1
    event_end_indices = np.where(diffs == -1)[0]

    # Handle case where signal is above threshold at the very beginning
    if above_thresh[0]:
        event_start_indices = np.insert(event_start_indices, 0, 0)
    # Handle case where signal remains above threshold at the very end
    if above_thresh[-1]:
        event_end_indices = np.append(event_end_indices, len(event_chan_data) - 1)

    # Map each event to its (start, end) sample indices
    events = {
        i + 1: (int(s), int(e))
        for i, (s, e) in enumerate(zip(event_start_indices, event_end_indices))
    }
    self.events = events

    # --- Extract peri‐event data for all recording channels ---
    n_events = len(event_start_indices)
    channels = self.recording_channels               # assumed zero‐based channel indices
    n_channels = len(channels)
    n_samples_per_event = int(self.pre_stimulus + self.post_stimulus)

    # Preallocate tensor: (events, channels, samples)
    event_tensor = np.zeros((n_events, n_channels, n_samples_per_event))

    for i, (start, _) in enumerate(zip(event_start_indices, event_end_indices)):
        # Compute absolute window bounds
        start_actual = int(start - self.pre_stimulus)
        end_actual = int(start + self.post_stimulus)
        for j, ch in enumerate(channels):
            # Slice raw_data for this event and channel
            event_window = self.raw_data[start_actual:end_actual, ch]
            event_tensor[i, j, :] = event_window

    self.mep_matrix = event_tensor

    # --- Optionally export ---
    if export:
        self._save_object(events, 'extracted_events', export_format)
        self._save_object(event_tensor, 'mep_matrix', export_format)

    return event_tensor




class Preprocessor:

    def __init__(self, trials, remove_bad_channels = False, compile_events = False):
        """
        Initializes the Preprocessor with parameters for hybrid distance calculation.
        """
        
        self.trials = trials
        self.remove = remove_bad_channels
        self.compile_events = compile_events

        self.bad_cutoff_percentile = 90

    def preprocess(self, export = True, output = None):

        if self.remove:

            progress = TqdmProgressBar()
            progress.run(self.trials, label="Removing Bad Channels", func=remove_bad_channels(alpha=0.5, beta=0.5, cutoff_percentile=self.bad_cutoff_percentile))

        if self.compile_events:


