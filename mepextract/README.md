# Motor Evoked Potentials (MEP) Extraction Package

## Overview

This Python package is designed to extract motor evoked potentials (MEPs) from Open Ephys data. It provides tools for loading raw electrophysiological data, detecting events, and extracting the corresponding MEPs for further analysis.

Please ensure you have the following packages installed:
- numpy
- open_ephys

You can install them using pip:

``` bash
pip install numpy open_ephys
```
## Usage
The extracted events and MEP data are automatically saved in the specified directory in JSON and Pickle formats, respectively.

1. Load raw electrophysiological data: Reads raw data from specified directories.
2. Event detection: Identifies events in the data based on a threshold.
3. MEP extraction: Extracts MEPs surrounding detected events.
4. Data saving: Saves extracted events and MEP data in JSON and Pickle formats.
  

Example workflow:

```python
from mepextract.src import Extractor

# initialise Extractor
extractor = Extractor(master_folder='path/to/master/folder', trial='trial_name')

# extract Raw Data
extractor.extract_raw()

# extract events and MEPs
events = extractor.extract_events(event_channel_number=_)
mep_data = extractor.get_event_data()
```


### Relevant methods/properties

__init__(self, master_folder, trial, pre=10, post=100, sampling_rate=30000)
- master_folder: Path to the master folder containing data.
    trial: Trial name.
- pre: Pre-stimulus time in milliseconds (default: 10 ms). 
- post: Post-stimulus time in milliseconds (default: 100 ms). 
- sampling_rate: Sampling rate in Hz (default: 30000 Hz).
extract_raw(self)
Loads the raw data from the specified directories and computes the data length.

**self.event_channel_number**: Pre-specified event channel

**self.data_length**: length of data from sample_numbers.npy

**self.raw_data**: extracted raw data

**self.events**: extracted events (in samples)

**self.mep**: extracted MEPs

**self.events**

## License
This project is licensed under the MIT License.

## Acknowledgments

Open Ephys for providing the data analysis tools.
Feel free to contact us if you have any questions or need further assistance.