# Motor Evoked Potentials (MEP) Extraction Package

## Overview

This Python package is designed to extract motor evoked potentials (MEPs) from Open Ephys data. It provides tools for loading raw electrophysiological data, detecting events, and extracting the corresponding MEPs for further analysis.

Please ensure you have the following packages installed:
- numpy
- pandas
- open_ephys
- matplotlib
- scipy
- json
- pickle
- logging

You can install them using pip:

``` bash
pip install numpy pandas open_ephys matplotlib scipy json pickle logging
```

## Usage

The extracted events and MEP data are automatically saved in the specified directory in JSON, MAT, and Pickle formats, respectively.

Load raw electrophysiological data: Reads raw data from specified directories.
Event detection: Identifies events in the data based on a threshold.
MEP extraction: Extracts MEPs surrounding detected events.
Data saving: Saves extracted events and MEP data in JSON, MAT, and Pickle formats.
Data plotting: Provides functionality for plotting raw data and extracted MEPs.

### Example workflow:

```python
from mepextract.extracting import Extractor

# initialise Extractor
extractor = Extractor(master_folder='path/to/master/folder', trial='trial_name', group='group_name', recording_channels=[1, 2, 3])

# extract Raw Data
extractor.extract_raw()

# extract events
events = extractor.extract_events(event_channel_number=1)

# extract MEP data
mep_data = extractor.get_event_data()

# plot raw data
extractor.plot_raw()

# plot average MEPs for specified channels
extractor.plot_event_average(channels=[1, 2, 3], amplitude_range=500)

# find and plot MEPs with specified peak parameters
peak_parameters = {'height': 0.5, 'distance': 10}
detected_peaks = extractor.find_meps(peak_parameters=peak_parameters)

# executes all of the above in one line
lazy(self, event_channel: int, peak_parameters: dict, export=True, plot=False, show=True)

# saves the current instance of the Extractor class to a file, excluding the raw_data so as
save_instance(self)


```


## Methods and Properties

### Properties

```python
init(self, master_folder, trial, group, recording_channels, pre=10, post=100, sampling_rate=30000)
    
    master_folder: # path to the master folder containing data
    trial: # trial name
    group: # group name
    recording_channels: # list of recording channels
    pre: # pre-stimulus time in milliseconds (default: 10 ms)
    post: # post-stimulus time in milliseconds (default: 100 ms)
    sampling_rate: # sampling rate in Hz (default: 30000 Hz)

```

`self.event_channel_number`: Pre-specified event channel.

`self.data_length`: Length of data from sample_numbers.npy.

`self.raw_data`: Extracted raw data.

`self.events`: Extracted events (in samples).

`self.mep`: Extracted MEPs.

### Method parameters

`event_channel_number`: The channel number to use for event extraction.

`export`: Whether to save the plot to a file (default: True).

`channels`: A list of channel numbers to include in the plot.

`amplitude_range`: The range of amplitude for the plot.

`save`: Whether to save the plot to a file (default: True).

`peak_parameters`: A dictionary containing parameters for peak detection.

`plot`: Whether to plot the data (default: True).

`show`: Whether to show the plots (default: False).
amplitude_range: The range of amplitude for the plot (default: 500).

`size`: The size of the plot (default: (21, 7)).

`export`: Whether to export the detected peaks (default: True).

## License

This project is licensed under the MIT License.