# openephysextract/__init__.py

# Core
from .session import Session

# Utilities
from .utilities import savify, loadify, spreadsheet, TqdmLoggingHandler

# Preprocessing pipeline + steps
from .preprocess import (
    Preprocessor,
    SessionStep,
    RemoveBadStep,
    DetrendStep,
    ASRStep,
    EOGRegressionStep,
    InterpolateStep,
    EventCompileStep,
    ReReferenceStep,
    FilterStep,
    SurfaceLaplacianStep,
    EpochStep,
    ArtifactRejectStep,
    DownsampleStep,
    ICARemovalStep,
)

# Extraction
from .extractor import Extractor

# Analysis
from .analysis import bandpower

__all__ = [
    # Core
    "Session",

    # Utilities
    "savify", "loadify", "spreadsheet", "TqdmLoggingHandler",

    # Preprocessing framework
    "Preprocessor",
    "SessionStep",

    # All defined preprocessing steps
    "RemoveBadStep",
    "DetrendStep",
    "ASRStep",
    "EOGRegressionStep",
    "InterpolateStep",
    "EventCompileStep",
    "ReReferenceStep",
    "FilterStep",
    "SurfaceLaplacianStep",
    "EpochStep",
    "ArtifactRejectStep",
    "DownsampleStep",
    "ICARemovalStep",

    # Extraction
    "Extractor",

    # Analysis
    "bandpower"
]
