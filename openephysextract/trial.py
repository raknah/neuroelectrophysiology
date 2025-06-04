import numpy as np
import pandas as pd

class Trial:
    def __init__(self, trial, data, sampling_rate=30000, source = None, location = None):

        '''Initializes a Trial object with trial number, data, and sampling rate.'''

        self.trial = trial
        self.data = data
        self.shape = data.shape
        self.notes = pd.Series(dtype=object)
        self.group = None
        self.sampling_rate = sampling_rate
        self.time_axis = np.arange(len(data)) / sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return f"Trial (#: {self.trial}, shape: {self.data.shape}, group: {self.group})"

    @property
    def duration(self):
        return len(self.data[0]) / self.sampling_rate

    @property
    def times(self):
        return np.arange(self.data.shape[1]) / self.sampling_rate

    def add_notes(self, notes):
        if isinstance(notes, dict):
            notes = pd.Series(notes)
        elif not isinstance(notes, pd.Series):
            raise TypeError("Notes must be a dictionary or a pandas Series.")
        self.notes = self.notes.append(notes, ignore_index=True)
        self.group = notes.get('group', self.group)
