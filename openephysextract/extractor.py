import numpy as np
import pandas as pd
import os
import json
import pickle

from open_ephys.analysis import Session

from .trial import Trial
from .progress import TqdmProgressBar  # Use the updated tqdm-based progress bar

class Extractor:
    def __init__(self, source, sampling_rate, channels = None, output=None):
        self.source = source
        self.files = os.listdir(source)
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.output = output if output else os.path.join(source, '000 output')

        # relevant paths
        self.path_to_all_data = 'Record Node 103/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data'

        # extracted attributes
        self.trials = []

    def extractify(self, n = None, export=False):
        """
        Extracts raw data from Open Ephys files and converts them into Trial objects.
        """

        if n is None:
            n = len(self.files)

        if not self.files:
            raise FileNotFoundError("No files found in the specified source directory.")

        self.trials = []

        def process_file(file):
            path = os.path.join(self.source, file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")

            sample_numbers = np.load(os.path.join(path, self.path_to_all_data, 'sample_numbers.npy'))
            data_length = sample_numbers.max() - sample_numbers.min()

            session = Session(path)
            recording = session.recordnodes[0].recordings[0]

            raw = recording.continuous[0].get_samples(
                start_sample_index=0,
                end_sample_index=data_length
            ).T

            if self.channels is None:
                self.channels = raw.shape[0]  # use all channels if none specified

            raw = raw[self.channels, :]

            if raw.shape[0] != len(self.channels):
                raise ValueError(f"Expected {len(self.channels)} channels, got {raw.shape[0]}")

            trial = Trial(file, raw, sampling_rate=self.sampling_rate)
            self.trials.append(trial)

        progress = TqdmProgressBar()
        progress.run(self.files[:n], label="Extracting Trials", func=process_file)

        if export:
            os.makedirs(self.output, exist_ok=True)
            with open(os.path.join(self.output, 'raw_data.pkl'), 'wb') as f:
                pickle.dump(self.trials, f)

        return self.trials