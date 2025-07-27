# Neuroelectrophysiology Framework 

A comprehensive, polyglot toolkit for extracting, processing, and analyzing neuroelectrophysiology data from Open Ephys recordings. This repository provides both **Python** and **Julia** implementations for maximum flexibility and performance in neuroscience research workflows.

## Purpose

This framework is designed for:
- **Motor Evoked Potential (MEP)** analysis and characterization
- **EEG/MEG signal processing** with advanced preprocessing pipelines
- **Cross-language compatibility** between Python and Julia ecosystems
- **High-performance analysis** leveraging Julia's speed and Python's ecosystem
- **Reproducible research** with version-controlled analysis pipelines

## Architecture

```
neuroelectrophysiology/
├── modules/
│   ├── openephysextract/          # Python package for data extraction & processing
│   │   ├── session.py             # Python Session class
│   │   ├── preprocess.py          # Preprocessing pipeline (PyTorch accelerated)
│   │   ├── analysis.py            # Spectral analysis tools
│   │   ├── extractor.py           # Open Ephys data extraction
│   │   └── pyproject.toml         # Python dependencies
│   └── sessionIO/                 # Julia package for high-performance analysis
│       ├── SessionIO.jl           # Julia Session struct
│       └── testSessionIO.jl       # Julia tests
├── data/                          # Shared experimental data (HDF5 format)
├── notebooks/                     # Analysis notebooks (Jupyter & Pluto)
├── scripts/                       # Utility scripts and converters
└── Project.toml                   # Julia project configuration
```

## Key Features

### Python Module (`openephysextract`)
- **Open Ephys Integration**: Direct extraction from Open Ephys recording sessions
- **GPU-Accelerated Processing**: PyTorch-based preprocessing pipeline with CUDA support
- **Advanced Preprocessing**: ASR, ICA, filtering, epoching, artifact rejection
- **Interactive Visualization**: Dash-based web interface for data exploration

### Julia Module (`SessionIO`)
- **High-Performance Loading**: Optimized HDF5 reading with proper memory layout
- **Type-Stable Operations**: Full type annotations for maximum Julia performance
- **Cross-Language Compatibility**: Seamlessly loads Python-generated HDF5 files
- **Memory Efficient**: Float32 arrays and column-major optimization
- **Functional Interface**: Immutable operations with convenient accessors

### Shared Data Format
- **HDF5 Standard**: Universal format readable by both languages
- **Rich Metadata**: JSON-encoded processing history, statistics, and annotations
- **Flexible Schema**: Supports raw, preprocessed, and epoched data arrays
- **Version Controlled**: Schema versioning for backward compatibility

## Installation

### Julia Environment
```julia
# From the repository root
julia --project=.
]activate .
]instantiate
```

### Python Environment
```bash
# Install the openephysextract package
cd modules/openephysextract
pip install -e .

# Or with conda
conda env create -f environment.yml
conda activate neuroelectrophysiology
```

## 🔬 Quick Start

### Loading Data in Julia
```julia
using Pkg; Pkg.activate(".")
include("modules/sessionIO/SessionIO.jl")

# Load a session
session = from_hdf5("data/2023-08-25_14-20-15.h5")

# Access data
println("Session: $(session.session)")
println("Shape: $(size(session))")
println("Duration: $(duration(session)) seconds")

# Index into epoched data
first_epoch = session[:, :, 1]  # All samples, all channels, first epoch
```

### Processing Data in Python
```python
from modules.openephysextract import Session, Preprocessor
from modules.openephysextract import FilterStep, EpochStep

# Load raw Open Ephys data
session = Session.from_open_ephys("path/to/recording")

# Create preprocessing pipeline
preprocessor = Preprocessor([
    FilterStep(lowcut=1.0, highcut=100.0),
    EpochStep(pre_stimulus=0.1, post_stimulus=0.5)
])

# Process and save
session = preprocessor.apply(session)
session.to_hdf5("processed_session.h5")
```

### Cross-Language Workflow
```python
# Python: Extract and preprocess
session = extract_open_ephys_data("raw_recording/")
session = preprocess_pipeline(session)
session.to_hdf5("processed.h5")
```

```julia
# Julia: Load and analyze
session = from_hdf5("processed.h5")
results = analyze_meps(session)
save_results(results, "analysis_output.h5")
```

## Data Format Specification

### HDF5 Structure
```
session.h5
├── /raw                    # (samples, channels) - Raw continuous data
├── /preprocessed           # (samples, channels) - Filtered/cleaned data  
├── /data                   # (samples_per_epoch, channels, epochs) - Epoched data
└── attributes:
    ├── session             # Session identifier (string)
    ├── experiment          # Experiment name (string)
    ├── sampling_rate       # Sampling frequency (int)
    ├── ch_names           # Channel names (JSON array)
    ├── history            # Processing steps (JSON array)
    ├── stats              # Analysis statistics (JSON object)
    └── events             # Event annotations (JSON array)
```

### Metadata Schema
```json
{
  "session": "2023-08-25_14-20-15",
  "experiment": "5xFAD Resting State",
  "sampling_rate": 30000,
  "ch_names": ["EMG1", "EMG2", "EEG1", "EEG2"],
  "history": [
    {"step": "filter", "params": {"lowcut": 1, "highcut": 100}, "time": "2023-08-25T14:21:00"},
    {"step": "epoch", "params": {"pre": 0.1, "post": 0.5}, "time": "2023-08-25T14:22:00"}
  ],
  "stats": {
    "epoch": {"n_epochs": 100, "rejected": 5},
    "filter": {"lowcut": 1.0, "highcut": 100.0}
  }
}
```


## Acknowledgments

- **Dr Nikolas Perentos**: For introducing me to neuroelectrophysiology and guiding me through the initial stages of this project
- **Open Ephys**: For the excellent open-source acquisition system
- **Julia Community**: For the high-performance scientific computing ecosystem  
- **Python Scientific Stack**: NumPy, SciPy, and the broader PyData ecosystem
- **HDF5 Group**: For the universal scientific data format

---


