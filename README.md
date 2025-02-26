# Motor Evoked Potentials (MEP) Analysis Package

## Overview
The **MEP Analysis** package (distributed as `mepextract`) provides tools for automated extraction, processing, and visualization of Motor Evoked Potentials (MEP) from electrophysiological recordings. Designed for scalability and flexibility, this package streamlines analysis workflows for large EEG datasets by offering:

- **Batch Processing**: Efficiently processes multiple trials with progress tracking.
- **Signal Extraction**: Leverages custom extraction methods to isolate MEP signals.
- **Visualization**: Utilizes high-quality plotting styles (integrating with `matplotlib` and `scienceplots`) to present analysis results.


### Features
- **Automated Trial Processing:** Loops through trials, skipping those marked for rejection.
- **Data Compatibility:** Reads input data from CSV files using `pandas`.
- **Custom Extraction Pipelines:** Uses the `Extractor` class to perform signal extraction.
- **Interactive Visualization:** Uses the `Viewer` class to render detailed plots of extracted signals and perform peak detection.
- **Modular Design:** Separates extraction (`extractor.py`) and visualization (`viewer.py`) into distinct modules for easy maintenance and extension.

## Usage

#### Import the Package
```python
from mepextract.extracting import Extractor, Viewer
import pandas as pd
import matplotlib.pyplot as plt
```

#### Load Your MEP metadata
The package expects information (saved as `Extractor.notes`) about sessions and trials (_session_name_, _phenotype_, _current level_, _stimulation type_) to be stored in a CSV file. Load this file using `pandas`:

```python
data_file = "path/to/your/MICE_MEP_2024.csv"
spreadsheet = pd.read_csv(data_file)
```

#### Process the Data
Extract signals from trials that are valid by looping through the trials and using the `Extractor` class:

```python
from mepextract.extracting import Extractor

sampling_rate = 30000  # Define your sampling rate (Hz)
all_extracted = []     # List to store extraction results

for i in range(len(spreadsheet)):
    if spreadsheet["sessionType"][i] == "reject":
        continue  # Skip trials marked as rejected
    else:
        extractor = Extractor(filepath=spreadsheet["filepath"][i],
                              sampling_rate=sampling_rate)
        extracted_data = extractor.extract_signal()
        all_extracted.append(extracted_data)
```
```python
# each entry in the list is a dictionary with the following keys
all_extracted[0].keys()
```

dict_keys(['trial', 'notes', 'events', 'data', 'event_mean', 'event_std'])

#### Visualize an Extracted Signal
Plot the extracted signal from the first valid trial:

```python
from mepextract.extracting import Viewer
viewer = Viewer(extracted = extracted, sampling_rate=30000)

# generate a GUI for peak detection and classification
viewer.classifier() 
```

```python
# create three lists based on classification (accepted, review and rejected) with the detected peaks appended
viewer.extract() 
accepted = viewer.accepted
review = viewer.review
rejected = viewer.rejected
```

```python
# each entry in the list is a dictionary with the following keys
accepted[0].keys()
```
dict_keys(['trial', 'notes', 'events', 'data', 'event_mean', 'event_std', 'positive peaks', 'negative peaks'])


## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to Dr. Nikolas Perentos for providing the initial data and insight for the development of this package.
