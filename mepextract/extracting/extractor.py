import numpy as np
import pandas as pd
import os
import json
import pickle
from open_ephys.analysis import Session
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d as gf
import scipy.io


class Extractor:
    def __init__(self, master_folder, trial, notes, recording_channels, pre=10, post=100, sampling_rate=30000):

        self.master_folder = master_folder
        self.notes = notes
        self.trial = trial
        self.sampling_rate = sampling_rate
        self.conversion = (1/sampling_rate)*1000
        self.pre_stimulus = (pre * (10 ** (-3))) * sampling_rate
        self.post_stimulus = (post * (10 ** (-3))) * sampling_rate
        self.num_channels = 16
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
        self.mep = None
        self.detected_peaks = {}


    # getter and setter for sampling_rate
    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        if value <= 0:
            raise ValueError("Sampling rate must be positive")
        self._sampling_rate = value


    # getter and setter for pre_stimulus
    @property
    def pre_stimulus(self):
        return self._pre_stimulus

    @pre_stimulus.setter
    def pre_stimulus(self, value):
        if value < 0:
            raise ValueError("Pre-stimulus time cannot be negative")
        self._pre_stimulus = value

    # getter and setter for post_stimulus
    @property
    def post_stimulus(self):
        return self._post_stimulus

    @post_stimulus.setter
    def post_stimulus(self, value):
        if value < 0:
            raise ValueError("Post-stimulus time cannot be negative")
        self._post_stimulus = value

    # getter and setter for number of channels
    @property
    def num_channels(self):
        return self._num_channels

    @num_channels.setter
    def num_channels(self, value):
        if value < 0:
            raise ValueError("Number of channels cannot be negative")
        self._num_channels = value

    @property
    def notes(self):
        return self.notes

    @notes.setter
    def notes(self, series):
        if not isinstance(series, pd.core.series.Series):
            raise ValueError("Notes must be a pandas series (pandas.core.series.Series)")
        self._notes = series

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

        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

    # event extraction function
    def extract_events(self, event_channel_number: int, export=True):
        """
        Extract event locations from the raw data based on the specified event channel.

        This method detects events in the specified event channel by identifying segments where the signal exceeds a
        predefined threshold. The start and end of each event are recorded and stored in the `events` attribute.
        Optionally, the detected events can be exported.

        Parameters:
            event_channel_number (int): The channel number to use for event extraction.
            export (bool, optional): Whether to export the detected events. Default is True.

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
            self._save_object(events, 'extracted_events', 'json')

        return events

    # extract MEPs
    def get_event_data(self, export=True):

        """
        Extract event data from the raw data.

        This method extracts data segments around each detected event for all channels. The segments include a pre-stimulus
        and post-stimulus period. The extracted data is stored in the `mep` attribute and can be optionally exported.

        Parameters:
            export (bool, optional): Whether to export the extracted event data. Default is True.

        Returns:
            numpy.ndarray: A 3D array containing the extracted event data for all channels.

        Raises:
            None
        """

        data = []

        for channel in range(0, 16):
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
            self._save_object(final_data, 'processed_data', 'pkl')
            self._save_object(final_data, 'processed_data', 'mat')
        return final_data

    # find MEPs and plot if desired
    def find_meps(self, peak_parameters: dict, temporal_range: tuple):
        """
        Find motor evoked potentials (MEPs) based on specified peak parameters.

        This method calculates the mean and standard deviation of the MEP data, finds positive and negative peaks,
        and optionally plots the results.

        The keys for peak_parameters can include:
        \n
            - 'height': Required height of peaks.
            - 'threshold': Required threshold of peaks.
            - 'distance': Required minimum horizontal distance (in samples) between neighboring peaks.
            - 'prominence': Required prominence of peaks.
            - 'width': Required width of peaks.
            - 'wlen': Used to calculate the width of the peaks.
            - 'rel_height': Relative height at which the peak width is measured.
            - 'plateau_size': Required size of the flat top of peaks.

        Parameters:
            peak_parameters (dict): A dictionary containing parameters for peak detection.
            plot (bool, optional): Whether to plot the data. Default is True.
            show (bool, optional): Whether to show the plots. Default is False.
            amplitude_range (int, optional): The range of amplitude for the plot. Default is 500.
            size (tuple, optional): The size of the plot. Default is (21, 7).
            export (bool, optional): Whether to export the detected peaks. Default is True.

        Returns:
            dict: A dictionary containing the detected peaks for each group.
            :param temporal_range:
        """

        if not isinstance(peak_parameters, dict):
            raise TypeError('''please enter the desired peak parameters as a dictionary in the following format {
            'height':, 'threshold':, 'distance':,  'prominence':, 'width':, 'wlen':, 'rel_height':, 'plateau_size': } ''')

        recording_channels = [channel - 1 for channel in self.recording_channels]

        search = dict(height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None,
                      rel_height=0.5, plateau_size=None)

        search.update(peak_parameters)

        for channel in recording_channels:

            temp = self.mep[channel, :, :]

            # baseline correction
            baseline = temp[:250, :].mean(axis=1).mean(axis=0)
            correction = np.full((1, temp.shape[0]), baseline)

            # mean
            mean_events = (temp.mean(axis = 1)).flatten()
            std_events = (temp.mean(axis = 1)).flatten()

            # smoothing with gaussian kernel of width ~5 samples
            smoothed_mean = gf(mean_events, sigma = 1.25)
            smoothed_std = gf(std_events, sigma = 1.25)

            # standardised data
            standardised = (smoothed_mean - smoothed_mean.mean())/smoothed_mean.std()

            # scaling by current
            standardised = (1/self.current)*standardised


            # find peaks and plot
            positive_peaks, positive_properties = find_peaks(
                mean_data[channel_index],
                height=search['height'],
                width=search['width'],
                distance=search['distance'],
                threshold=search['threshold'],
                prominence=search['prominence'],
                wlen=search['wlen'],
                rel_height=search['rel_height'],
                plateau_size=search['plateau_size']
            )
            negative_peaks, negative_properties = find_peaks(
                -mean_data[channel_index],
                height=search['height'],
                width=search['width'],
                distance=search['distance'],
                threshold=search['threshold'],
                prominence=search['prominence'],
                wlen=search['wlen'],
                rel_height=search['rel_height'],
                plateau_size=search['plateau_size']
            )

            for peak in range(len(positive_peaks)):
                if positive_peaks[peak] > self.pre_stimulus:
                    self.detected_peaks[self.group].append(
                        (int(positive_peaks[peak]), int(positive_properties['peak_heights'][peak])))

            for peak in range(len(negative_peaks)):
                if negative_peaks[peak] > self.pre_stimulus:
                    self.detected_peaks[self.group].append(
                        (int(negative_peaks[peak]), int(negative_properties['peak_heights'][peak])))

        self._save_object(self.detected_peaks, 'detected_peaks', 'json')



        return self.detected_peaks

    def lazy(self, event_channel: int, peak_parameters: dict, export=True, plot=False, show=True):
        """
        Perform a series of data extraction and processing steps in a lazy manner.

        This method sequentially calls the following methods:
            1. `extract_raw`: Extracts raw data from the specified source.
            2. `extract_events`: Extracts event locations from the raw data based on the specified event channel.
            3. `get_event_data`: Extracts event data from the raw data.
            4. `find_meps`: Finds motor evoked potentials (MEPs) based on the specified peak parameters.
        Parameters:
            event_channel (int): The channel number to use for event extraction.
            peak_parameters (dict): A dictionary containing parameters for peak detection.
            export (bool, optional): Whether to export the extracted data. Default is True.
            plot (bool, optional): Whether to plot the data. Default is False.
            show (bool, optional): Whether to show the plots. Default is True.

        Returns:
            detected_peaks
        """

        self.extract_raw()
        self.extract_events(event_channel_number=event_channel, export=export)
        self.get_event_data(export=export)
        self.find_meps(peak_parameters=peak_parameters, plot=plot, show=True, export=export)

    # save Extractor object to external location
    def save_instance(self):
        """
        Save the current instance of the Extractor class to a file, excluding the raw_data attribute.

        The instance is saved using pickle with the filename based on the trial attribute.

        Parameters:
            None

        Returns:
            None
        """
        raw_data_backup = self.raw_data
        self.raw_data = None  # Temporarily set raw_data to None

        file_name = f"{self.trial}.pkl"
        full_path = os.path.join(self.path_for_extracted, file_name)

        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

        if self.log:
            self.logger.info(f"Instance saved as {file_name} (excluding raw_data)")

        self.raw_data = raw_data_backup  # Restore raw_data

