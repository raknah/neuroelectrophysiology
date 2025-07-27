import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from open_ephys.analysis import Session as OpenEphysSession
from tqdm import tqdm
from .session import Session

# Configure logging for extraction process
logger = logging.getLogger(__name__)

class Extractor:
    """
    Extractor for loading raw electrophysiology data into Session objects.
    """

    def __init__(
            self,
            source: str,
            experiment: str,
            sampling_rate: float,
            output: str,
            notes: Optional[dict] = None,
            channels: Optional[List[int]] = None,
            path_to_all_data: str = (
                    'Record Node 103/experiment1/recording1/continuous/' +
                    'OE_FPGA_Acquisition_Board-100.Rhythm Data'
            )
    ):
        self.source = source
        self.experiment = experiment
        self.files = sorted(os.listdir(source))
        self.notes = notes.set_index('Session').to_dict(orient='index') if isinstance(notes, pd.DataFrame) else notes or {}
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.output = output
        self.path_to_all_data = path_to_all_data
        self.sessions: List[Session] = []

    def extractify(
            self,
            n: Optional[int] = None,
            export: bool = False,
            include: Optional[List[str]] = None
    ) -> List[Session]:
        """
        Extract raw data into Session objects.
        """
        if include is not None:
            files = [f for f in include if f in self.files]
            if not files:
                raise ValueError("None of the requested sessions found in directory.")
        else:
            files = self.files if n is None else self.files[:n]

        self.sessions = []

        def _process(fname: str):
            folder = os.path.join(self.source, fname)
            if not os.path.isdir(folder):
                logger.warning(f"Skipping non-directory: {folder}")
                return

            data_dir = os.path.join(folder, self.path_to_all_data)
            sample_numbers_path = os.path.join(data_dir, 'sample_numbers.npy')
            if not os.path.exists(sample_numbers_path):
                raise FileNotFoundError(f"Missing sample_numbers.npy at: {sample_numbers_path}")

            sample_numbers = np.load(sample_numbers_path)
            nsamp = int(sample_numbers.max() - sample_numbers.min())

            # raw_path = os.path.join(data_dir, 'continuous.dat')
            # if not os.path.exists(raw_path):
            #     raise FileNotFoundError(f"Missing continuous.dat at: {raw_path}")

            open_ephys_sesh = OpenEphysSession(os.path.join(self.source, fname))
            recording = open_ephys_sesh.recordnodes[0].recordings[0]
            data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=nsamp).T

            raw = data if self.channels is None else data[self.channels, :]

            if self.channels and raw.shape[0] != len(self.channels):
                raise ValueError(
                    f"Channel mismatch for {fname}: "
                    f"expected {len(self.channels)}, got {raw.shape[0]}"
                )

            session = Session(
                session=fname,
                experiment=self.experiment,
                sampling_rate=self.sampling_rate,
                raw=raw,
                preprocessed=None,
                data=None,
                location=folder
            )
            if self.notes:
                session.add_notes(self.notes.get(fname, {}))

            self.sessions.append(session)

        for file in tqdm(files, desc="Extracting Sessions", unit="file"):
            _process(file)

        if export:
            os.makedirs(self.output, exist_ok=True)
            out_path = os.path.join(self.output, f"{self.experiment} RAW.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(self.sessions, f)
            logger.info(f"Exported {len(self.sessions)} sessions to {out_path}")

        return self.sessions
