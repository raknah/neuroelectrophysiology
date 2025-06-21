# Open Ephys Extract

## Overview
`openephysextract` offers a streamlined workflow for loading, cleaning and exploring data recorded with the [Open Ephys](https://open-ephys.org/) system. The package converts raw acquisitions into easy-to-handle `Trial` objects and exposes utilities for preprocessing, interactive visualisation and basic analysis.

Key components include:

- **Extractor** – reads continuous Open Ephys folders and builds `Trial` objects containing the raw samples.
- **Preprocessor** – applies modular steps (bad channel removal, filtering, event compilation, epoching and downsampling) to a list of trials.
- **Viewer** – Jupyter widget for browsing trials, detecting MEP peaks and classifying each trial as accepted, review or rejected.
- **Analysis & Plotting** – convenience functions for band power extraction, logistic scaling and displaying raw signals, spectrograms and PSDs.

The examples below demonstrate a minimal workflow.

## Quickstart
Below is a condensed version of the workflow found in the *5xFAD Resting State EEG* notebook.
It demonstrates how to configure destination folders for extracted and processed data.

```python
import os
import pickle

from openephysextract import Extractor, Preprocessor, Viewer
from openephysextract.preprocess import (
    RemoveBadStep, DownsampleStep, FilterStep, EpochStep,
)
from openephysextract.analysis import bandpower, logistic_scaler
from openephysextract.utilities import savify

# ---- paths ----
source = "/path/to/open_ephys_sessions"  # folder with session directories
channels = [3, 4, 5, 6, 7, 8]             # indices of data channels to load
sampling_rate = 30000                     # original acquisition rate in Hz

# folders used by the pipeline
output_raw = os.path.join(source, "processed")      # extracted Trials saved here
output_pp = "/path/to/resting-state-analysis"       # destination for preprocessing outputs

# ---- extraction ----
extractor = Extractor(
    source=source,           # base folder of Open Ephys files
    channels=channels,       # channel indices to include
    sampling_rate=sampling_rate,  # sampling rate of recordings
    output=output_raw,       # where to store raw Trial objects
)
trials = extractor.extractify(export=True)  # returns a list of Trial objects

# reload the pickled trials if needed
with open(os.path.join(output_raw, "raw_data.pkl"), "rb") as f:
    trials = pickle.load(f)

# ---- preprocessing ----
steps = [
    RemoveBadStep(alpha=0.3, beta=0.7),  # remove noisy channels
    DownsampleStep(target_fs=300),       # decimate to 300 Hz
    FilterStep(lowcut=0.1, highcut=80, order=4),  # band-pass filter
    EpochStep(frame=300, stride=30),     # segment into 300‑sample windows
]
processor = Preprocessor(
    trials=trials,
    steps=steps,
    destination=output_pp,   # where processed trials will be saved
)
processed = processor.preprocess(parallel=False, export=True)

# ---- visualise ----
Viewer(processed, sampling_rate=300).classifier()  # interactive labelling widget

# ---- analysis ----
features = [bandpower(trial) for trial in processed]  # extract spectral power
savify(features, output_pp, "features")

scaled = [logistic_scaler(trial) for trial in features]  # logistic normalisation
savify(scaled, output_pp, "logistic-scaled")
```

## Installation
The project requires Python 3.8+. Install directly from the repository:
```bash
pip install git+https://github.com/raknah/motor-evoked-potentials-analysis
```

## License
Released under the MIT license.

## Acknowledgments
Special thanks to Dr. Nikolas Perentos for providing the initial data and insight for the development of this package.
