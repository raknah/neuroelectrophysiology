import numpy as np
import os
import json
import pickle
from open_ephys.analysis import Session


class Extractor:
    def __init__(self, master_folder, trial, pre=10, post=100, sampling_rate=30000):
        self.master_folder = master_folder
        self.trial = trial
        self.sampling_rate = sampling_rate
        self.pre_stimulus = (pre*(10**(-3)))*30000
        self.post_stimulus = (post*(10**(-3)))*30000

        # relevant paths
        self.path_to_all_data = 'Record Node 103/experiment1/recording1/continuous/OE_FPGA_Acquisition_Board-100.Rhythm Data'
        self.path_to_sample_numbers = os.path.join(self.master_folder, self.trial, self.path_to_all_data, 'sample_numbers.npy')
        self.path_to_raw_data = os.path.join(self.master_folder, self.trial)
        self.path_for_extracted = os.path.join(self.master_folder, self.trial, 'extracted')
        if not os.path.exists(self.path_for_extracted):
            os.makedirs(self.path_for_extracted)

        # attributes
        self.data_length = None
        self.raw_data = None
        self.events = None
        self.mep = None

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

    # save extracted objects
    def _save_object(self, object, name, type):
        path = self.path_for_extracted
        if type == 'pkl':
            file_name = name + ".pkl"
            full_path = os.path.join(path, file_name)
            with open(full_path, 'wb') as f:
                pickle.dump(object, f)
        elif type == 'json':  # Use elif instead of if to avoid checking both conditions if the first one is true
            file_name = name + ".json"
            full_path = os.path.join(path, file_name)
            with open(full_path, 'w') as f:
                json.dump(object, f)
        else:
            raise ValueError("Unsupported file type. Please use 'pkl' or 'json'.")

    # extract raw data
    def extract_raw(self):
        # load sample numbers
        sample_numbers = np.load(self.path_to_sample_numbers)
        self.data_length = sample_numbers.max() - sample_numbers.min()
        # load raw data
        session = Session(self.path_to_raw_data)
        recording = session.recordnodes[0].recordings[0]
        self.raw_data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=self.data_length)

    # event extraction function
    def extract_events(self, event_channel_number):
        data = self.raw_data
        threshold = 0.5  # threshold for detecting events which are always above 0.5
        event_channel_data = data[:, event_channel_number - 1]

        events = {}  # initialize a dictionary to store events with unique identifiers
        event_number = 1  # start naming events from 1

        is_event = False  # initialize a flag to track if an event is ongoing
        event_start = None
        event_end = None

        for index, value in enumerate(event_channel_data):
            if value >= threshold:
                if not is_event:
                    event_start = index  # record the start of a new event
                    is_event = True
            else:
                if is_event:
                    event_end = index - 1  # record the end of the ongoing event
                    events[event_number] = (event_start, event_end)  # store the event
                    event_number += 1  # increment event name for the next event
                    is_event = False  # reset the event flag

        # check if an event is ongoing at the end of the channel
        if is_event:
            events[event_number] = (event_start, len(event_channel_data - 1))  # handle the final ongoing event

        self.events = events
        self._save_object(events, 'extracted_events', 'json')
        return events

    # extract MEPs
    def get_event_data(self):

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
        self.mep = final_data
        self._save_object(final_data, 'processed_data', 'pkl')
        return np.array(data)