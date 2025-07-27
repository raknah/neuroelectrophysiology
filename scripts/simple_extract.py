#!/usr/bin/env python3
"""
Simple 5xFAD Processing - Direct from Notebook Workflow

A minimal script that replicates the exact notebook workflow:
1. Extract sessions
2. Preprocess with standard pipeline  
3. Save to HDF5

Edit the paths below and run: python simple_extract.py
"""

import sys
import os
from pathlib import Path

# Add the modules directory to Python path
script_dir = Path(__file__).parent.parent  # Go up to neuroelectrophysiology/
modules_dir = script_dir / "modules"
sys.path.insert(0, str(modules_dir))

# Verify the path exists
if not modules_dir.exists():
    raise FileNotFoundError(f"Modules directory not found: {modules_dir}")

from openephysextract.extractor import Extractor
from openephysextract.preprocess import (
    Preprocessor, RemoveBadStep, FilterStep, 
    DownsampleStep, EpochStep, StandardizeStep
)
from openephysextract.utilities import spreadsheet

# ================================
# EDIT THESE PATHS FOR YOUR SETUP
# ================================
SOURCE_FOLDER = '/Volumes/STORAGE 1.0/UNIC Research/5xFAD Resting State'
OUTPUT_FOLDER = '/Users/fomo/Documents/Research/UNIC Research/Neuroelectrophysiology/5xFAD Resting State/data'
NOTES_PATH = '/Users/fomo/Documents/Research/UNIC Research/Neuroelectrophysiology/notes/MICE_LIST_EEG.xlsx'

def main():
    print("🧠 5xFAD Simple Processing Pipeline")
    print("=" * 40)
    
    # Step 1: Load notes (or create minimal version)
    try:
        notes = spreadsheet(
            location='/Users/fomo/Documents/Research/UNIC Research/Neuroelectrophysiology/notes',
            name='MICE_LIST_EEG.xlsx',
            id='Session',
            relevant=sorted(os.listdir(SOURCE_FOLDER)),
            sheet='MEP'
        )
        print(f"📋 Loaded notes for {len(notes)} sessions")
    except:
        print("📋 Using session names as notes")
        sessions = sorted(os.listdir(SOURCE_FOLDER))
        notes = None
    
    # Step 2: Extract sessions
    print("🔬 Extracting sessions...")
    extractor = Extractor(
        source=SOURCE_FOLDER,
        experiment='5xFAD Resting State',
        sampling_rate=30000,
        output='/tmp/5xfad_cache',
        notes=notes,
        channels=[3, 4, 5, 6, 7, 8]  # S1 L+R, V1/V2 L+R
    )
    
    sessions = extractor.extractify(export=False)
    print(f"✅ Extracted {len(sessions)} sessions")
    
    # Step 3: Create preprocessing pipeline (exact notebook settings)
    print("⚙️ Setting up preprocessing...")
    steps = [
        RemoveBadStep(std=True, alpha=0.5, beta=0.5, cutoff_pct=90),
        FilterStep(lowcut=0.1, highcut=80, order=4),
        DownsampleStep(target_fs=100, downsample_raw=True),
        EpochStep(frame=100, stride=10),  # 1s epochs, 90% overlap
        StandardizeStep(method='zscore', per_epoch=True)
    ]
    
    preprocessor = Preprocessor(steps=steps, device='mps', verbose=True)
    
    # Step 4: Preprocess sessions
    print("🔄 Preprocessing sessions...")
    processed_sessions = preprocessor.preprocess(sessions, use_gpu=True)
    print(f"✅ Preprocessed {len(processed_sessions)} sessions")
    
    # Step 5: Save to HDF5
    print("💾 Saving to HDF5...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for session in processed_sessions:
        output_path = f"{OUTPUT_FOLDER}/{session.session}.h5"
        session.to_hdf5(output_path)
        print(f"💾 Saved: {session.session}.h5")
    
    print("🎉 Processing complete!")
    print(f"📁 Files saved to: {OUTPUT_FOLDER}")
    print("\n🔬 Load in Julia with:")
    print("session = from_hdf5(\"path/to/session.h5\")")

if __name__ == "__main__":
    main()
