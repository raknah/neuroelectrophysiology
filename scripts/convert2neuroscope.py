#!/usr/bin/env python3
"""
convert_to_neuroscope.py

Convert a NumPy array of EEG data into Neuroscope-2 compatible files:
  • .eeg  — raw int16 binary (channels × samples)
  • .xml  — metadata (sampling rate, #channels, resolution)
  • .evt  — optional event file marking start/end & a label

Usage:
  ./convert_to_neuroscope.py \
    --input  session01.npy \
    --output-dir ./converted \
    --samplerate 1250 \
    --scale    1000 \
    --label    WT
"""

import os
import argparse
import numpy as np

def save_eeg_binary(data, out_file, scale):
    """
    Save data (time × channels) as little-endian int16 .eeg file.
    scale: value to multiply data by before converting to int16.
    """
    # scale, convert to int16, transpose to (channels, time), then write
    data_int16 = (data * scale).astype('<i2')
    data_int16.T.tofile(out_file)

def write_xml_metadata(out_file, n_channels, samplerate):
    """
    Write a minimal Neuroscope-2 XML metadata file.
    """
    xml = f"""
    <?xml version="1.0"?>
    <parameters>
      <acquisitionSystem>
        <nChannels>{n_channels}</nChannels>
        <samplingRate>{samplerate}</samplingRate>
        <voltageRange>5000</voltageRange>
        <resolution>16</resolution>
        <amplification>1</amplification>
      </acquisitionSystem>
      <fieldPotentials>
        <lfpSamplingRate>{samplerate}</lfpSamplingRate>
      </fieldPotentials>
    </parameters>
"""
    with open(out_file, 'w') as f:
        f.write(xml)

def write_evt_file(out_file, n_samples, label):
    """
    Write an .evt file marking start (sample 0) and end (n_samples),
    with a text label for the session or group.
    """
    with open(out_file, 'w') as f:
        f.write(f"0\tStart: {label}\n")
        f.write(f"{n_samples}\tEnd\n")

def main():
    p = argparse.ArgumentParser(
        description="Convert EEG (NumPy .npy) to Neuroscope-2 .eeg/.xml/.evt"
    )
    p.add_argument('--input',       '-i',
                   required=True,
                   help="Input .npy file containing EEG array (time × channels)")
    p.add_argument('--output-dir',  '-o',
                   required=True,
                   help="Directory to write .eeg, .xml, (and .evt)")
    p.add_argument('--samplerate',  '-s',
                   type=int, default=1250,
                   help="Sampling rate (Hz), default 1250")
    p.add_argument('--scale',       '-c',
                   type=float, default=1000.0,
                   help="Scale factor before int16 conversion, default 1000")
    p.add_argument('--label',       '-l',
                   help="Optional label for .evt (e.g. WT or 5xFAD)")
    args = p.parse_args()

    # Load data
    data = np.load(args.input)
    if data.ndim != 2:
        p.error("Input array must be 2D: (timepoints, channels)")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    # Write .eeg
    eeg_path = os.path.join(args.output_dir, base + '.eeg')
    save_eeg_binary(data, eeg_path, args.scale)
    print(f"Wrote EEG binary → {eeg_path}")

    # Write .xml
    xml_path = os.path.join(args.output_dir, base + '.xml')
    write_xml_metadata(xml_path, n_channels=data.shape[1], samplerate=args.samplerate)
    print(f"Wrote XML metadata → {xml_path}")

    # Write .evt if requested
    if args.label:
        evt_path = os.path.join(args.output_dir, base + '.evt')
        write_evt_file(evt_path, n_samples=data.shape[0], label=args.label)
        print(f"Wrote event file → {evt_path}")

if __name__ == '__main__':
    main()
