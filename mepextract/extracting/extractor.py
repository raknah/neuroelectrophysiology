import numpy as np
import os
import json
import pickle
from open_ephys.analysis import Session
import scipy.io


class Extractor:
    def __init__(self, master_folder, notes, recording_channels, sampling_rate, pre_stimulus_ms=10, post_stimulus_ms=100):

        self.master_folder = master_folder
        self.notes = notes
        self.trial = notes['session']
        self.sampling_rate = sampling_rate
        self.conversion = (1/sampling_rate)*1000
        self.pre_stimulus = (pre_stimulus_ms * (10 ** (-3))) * sampling_rate
        self.post_stimulus = (post_stimulus_ms * (10 ** (-3))) * sampling_rate
        self.num_channels = None
        self.recording_channels = recording_channels

        # relevant paths
        self.path_to_all_data = 'Record Node 103/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data'
        self.path_to_sample_numbers = os.path.join(self.master_folder, self.trial, self.path_to_all_data,
                                                   'sample_numbers.npy')
        self.path_to_raw_data = os.path.join(self.master_folder, self.trial)
        self.path_for_extracted = os.path.join(self.master_folder, self.trial, 'extracted')
        if not os.path.exists(self.path_for_extracted):
            os.makedirs(self.path_for_extracted)

        # extracted attributes
        self.data_length = None
        self.raw_data = None
        self.events = None
        self.mep_matrix = None

    # (internal function) save extracted objects
    def _save_object(self, pyobject, name, datatype):

        """
        Save a Python object to a file in the specified format.

        This method saves the given Python object to a file in one of the supported formats: pickle, JSON, or MAT.
        The file is saved in the directory specified by `self.path_for_extracted`.

        Parameters:
            pyobject (object): The Python object to be saved.
            name (str): The name of the file (without extension) to save the object to.
            datatype (str): The format to save the file in. Supported formats are 'pkl', 'json', and 'mat'.

        Returns:
            None

        Raises:
            ValueError: If `datatype` is not one of the supported formats ('pkl', 'json', 'mat').
        """

        path = self.path_for_extracted

        if datatype == 'pkl':
            file_name = name + ".pkl"
            full_path = os.path.join(path, file_name)
            with open(full_path, 'wb') as f:
                pickle.dump(pyobject, f)
        elif datatype == 'json':  # Use elif instead of if to avoid checking both conditions if the first one is true
            file_name = name + ".json"
            full_path = os.path.join(path, file_name)
            with open(full_path, 'w') as f:
                json.dump(pyobject, f)
        elif datatype == 'mat':
            file_name = name + ".mat"
            full_path = os.path.join(path, file_name)
            scipy.io.savemat(full_path, {name: pyobject})
        else:
            raise ValueError("Unsupported file type. Please use 'pkl', 'json' or 'mat.")

    # extract raw data
    def extract_raw(self):

        """
        Extract raw data from the specified source.

        This method loads sample numbers and raw data from the specified paths, and stores the raw data in the `raw_data` attribute.
        The length of the data is calculated based on the sample numbers.

        Parameters:
            None

        Returns:
            None

        Raises:
            FileNotFoundError: If the sample numbers file or raw data path does not exist.
        """

        # load sample numbers
        try:
            # load sample numbers
            sample_numbers = np.load(self.path_to_sample_numbers)
            self.data_length = sample_numbers.max() - sample_numbers.min()

            # load raw data
            session = Session(self.path_to_raw_data)
            recording = session.recordnodes[0].recordings[0]
            self.raw_data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=self.data_length)
            self.num_channels = self.raw_data.shape[0]

        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

    # event extraction function
    def extract_events(self, event_channel_number: int, export=True, export_format = 'pkl'):
        """
        Extract event locations from the raw data based on the specified event channel.

        This method detects events in the specified event channel by identifying segments where the signal exceeds a
        predefined threshold. The start and end of each event are recorded and stored in the `events` attribute.
        Optionally, the detected events can be exported.

        Parameters:
            event_channel_number (int): The channel number to use for event extraction.
            export (bool, optional): Whether to export the detected events. Default is True.
            export_format: The format to save the exported events in. Supported formats are 'pkl', 'json', and 'mat'.

        Returns:
            dict: A dictionary containing the detected events with their start and end sample indices.

        Raises:
            TypeError: If `event_channel_number` is not an integer.
            ValueError: If `event_channel_number` is not within the valid range of channels.
        """

        if not isinstance(event_channel_number, int):
            raise TypeError('event_channel_number must be an integer')

        if event_channel_number < 1 or event_channel_number > self.num_channels:
            raise ValueError('event_channel_number must be within the valid range of channels')

        data = self.raw_data
        threshold = 0.5  # threshold for detecting events which are always above 0.5
        event_channel_data = data[:, event_channel_number - 1]

        # Detect event locations using vectorized operations
        above_threshold = event_channel_data >= threshold
        event_start_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
        event_end_indices = np.where(np.diff(above_threshold.astype(int)) == -1)[0]

        # Handle the case where an event starts but does not end within the data
        if above_threshold[0]:
            event_start_indices = np.insert(event_start_indices, 0, 0)
        if above_threshold[-1]:
            event_end_indices = np.append(event_end_indices, len(event_channel_data) - 1)

        events = {i + 1: (int(start), int(end)) for i, (start, end) in enumerate(zip(event_start_indices, event_end_indices))}

        self.events = events

        if export:
            self._save_object(events, 'extracted_events', export_format)


    # extract MEPs
    def get_event_data(self, export=True, export_format='pkl'):

        """
        Extract event data from the raw data.

        This method extracts data segments around each detected event for all channels. The segments include a pre-stimulus
        and post-stimulus period. The extracted data is stored in the `mep` attribute and can be optionally exported.

        Parameters:
            export (bool, optional): Whether to export the extracted event data. Default is True.
            export_format: The format to save the exported event data in. Supported formats are 'pkl', 'json', and 'mat'.

        Returns:
            numpy.ndarray: A 3D array containing the extracted event data for all relevant channels.

        Raises:
            None
        """

        data = []

        for channel in self.recording_channels:
            extracted_data = []
            for event_number, (start, end) in self.events.items():
                start_actual = int(start - self.pre_stimulus)
                end_actual = int(start + self.post_stimulus)
                event_data = self.raw_data[start_actual:end_actual, channel]
                extracted_data.append(event_data)

            data.append(np.vstack(extracted_data).T)

        final_data = np.array(data)
        self.mep_matrix = final_data
        if export:
            self._save_object(final_data, 'mep_matrix', export_format)

    def lazy(self, event_channel: int, export=True, export_format='pkl'):
        """
        Perform a series of data extraction and processing steps in a lazy manner.

        This method sequentially calls the following methods:
            1. `extract_raw`: Extracts raw data from the specified source.
            2. `extract_events`: Extracts event locations from the raw data based on the specified event channel.
            3. `get_event_data`: Extracts event data from the raw data.

        Parameters:
            event_channel (int): The channel number to use for event extraction.
            export (bool, optional): Whether to export the extracted data. Default is True.
            export_format: The format to save the exported data in. Supported formats are 'pkl', 'json', and 'mat'.

        """

        self.extract_raw()
        self.extract_events(event_channel_number=event_channel, export=export, export_format=export_format)
        self.get_event_data(export=export, export_format=export_format)

        return {'trial': self.trial, 'notes': self.notes, 'data': self.mep_matrix}