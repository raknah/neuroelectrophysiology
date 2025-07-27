# OpenEphys Extract

A Python package for convenient extraction and analysis of OpenEphys data, with a focus on motor-evoked potentials and resting state analysis.

## Features

- **Data Extraction**: Load and handle OpenEphys recording data
- **Preprocessing Pipeline**: Comprehensive preprocessing tools listed below
- **Analysis Tools**: Including bandpower analysis and state-space modeling
- **Utility Functions**: Save/load operations, spreadsheet handling
- **Easy to Use**: Simple API for quick integration into your analysis workflows

Requires Python 3.8 or later.

## Quick Start

```python
from openephysextract import Session, Preprocessor

# Load a session
session = Session(
                session="session_name",
                experiment="experiment_name",
                source="data_path",
                sampling_rate=30000,
                ch_names= [2,4,5,9],
            )

# Create a preprocessing pipeline
preprocessor = Preprocessor([
    "RemoveBadStep",
    "FilterStep",
    "EpochStep"
])

# Process your data
processed_data = preprocessor.run(session)
```
## Resting State Analysis Workflow

Here's a typical workflow for resting state analysis:

1. **Data Extraction** from OpenEphys Recordings
```python
from openephysextract import Extractor

extractor = Extractor(
    source="data_path",
    experiment="experiment_name",
    sampling_rate=30000,
    output="output_path",
    channels=[3, 4, 5, 6, 7, 8]  # Example channels
)

raw_data = extractor.extractify(export=True)
```
2. **Preprocessing Pipeline**
```python
from openephysextract.preprocess import (
    Preprocessor, RemoveBadStep, FilterStep, 
    DownsampleStep, EpochStep
)

steps = [
    RemoveBadStep(std=True, alpha=0.5, beta=0.5, cutoff_pct=90),
    FilterStep(lowcut=0.1, highcut=80, order=4),
    DownsampleStep(target_fs=100),
    EpochStep(frame=100, stride=10)  # 1s epochs with 90% overlap
]

preprocessor = Preprocessor(steps=steps)
preprocessed = preprocessor.preprocess(raw_data)
```
3. **Analysis**

   - Bandpower analysis
   - State-space modeling (e.g., Beta-HMM for dynamical analysis)
   - Statistical comparisons between conditions

## Core Components

### Session
The `Session` class is your entry point for working with OpenEphys data. It handles data loading and provides a consistent interface for the preprocessing pipeline.

### Preprocessor
The `Preprocessor` class manages the preprocessing pipeline. You can chain multiple preprocessing steps together to create your desired workflow.

#### Session Step
You can also use the `SessionStep` to define custom preprocessing steps to a session object, allowing for a more personalised workflow.

```python
class SessionStep(ABC):
    """Abstract preprocessing step for EEG sessions."""
    @abstractmethod
    def apply(self, session: Session, device: torch.device) -> None:
        """Mutate session.raw/preprocessed/data on `device`."""
        ...

    def preferred_device(self, default: torch.device) -> torch.device:
        return default  # override in subclasses as needed
```

### Extractor
The `Extractor` class provides tools for extracting specific data segments and features from your processed sessions.

## Available Preprocessing Steps

- `RemoveBadStep`: Remove bad channels
- `DetrendStep`: Remove signal trends
- `ASRStep`: Artifact Subspace Reconstruction
- `EOGRegressionStep`: EOG artifact regression
- `InterpolateStep`: Channel interpolation
- `EventCompileStep`: Event handling
- `ReReferenceStep`: Signal re-referencing
- `FilterStep`: Signal filtering
- `SurfaceLaplacianStep`: Surface Laplacian transformation
- `EpochStep`: Data epoching
- `ArtifactRejectStep`: Artifact rejection
- `DownsampleStep`: Signal downsampling
- `ICARemovalStep`: ICA-based artifact removal

## Utility Functions

- `savify`: Save processed data
- `loadify`: Load saved data
- `spreadsheet`: Spreadsheet operations
- `bandpower`: Compute signal bandpower

## License

This project is licensed under the MIT License.

## Author

Aswinshankar Sivalingam

## Acknowledgments

This package is built on the foundation of OpenEphys data handling and analysis, leveraging existing libraries and tools for efficient processing and analysis of electrophysiological data. 

I'd also like to thank Dr. Perentos for his guidance and support in understanding the neuroelectrophysiology workflows that led to the development of this package.
```