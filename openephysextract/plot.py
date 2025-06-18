import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch

def plotifyRAWdata(
        trial,
        nperseg=64,
        noverlap=32,
        figsize=(30, 10),
        dpi=210
):
    """
    Plot raw EEG, time-frequency spectrogram of the entire session, and PSD for a Trial.

    Parameters
    ----------
    trial : Trial
        Trial object with .raw (channels x samples) and .data (epochs x channels x samples).
    nperseg : int
        Length of each segment for spectrogram and Welch PSD.
    noverlap : int
        Number of points to overlap between segments.
    figsize : tuple
        Figure size in inches (width, height).
    dpi : int
        Figure dots-per-inch resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : dict of matplotlib.axes.Axes
        Dictionary with keys 'orig', 'tf', 'psd'.
    """
    fs = trial.sampling_rate

    # Create mosaic layout
    fig, axes = plt.subplot_mosaic(
        [['orig', 'psd'],
         ['tf',   'psd']],
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={
            'width_ratios': [4, 1],
            'height_ratios': [1.5, 1]
        }
    )

    # 1) Plot raw continuous EEG on 'orig'
    raw = trial.raw  # shape: (n_channels, n_samples)
    times = np.arange(raw.shape[1]) / fs
    for ch in range(raw.shape[0]):
        axes['orig'].plot(times, raw[ch], label=f'ch{ch}')
    axes['orig'].set_title('Raw Continuous EEG')
    axes['orig'].set_xlabel('Time (s)')
    axes['orig'].set_ylabel('Amplitude (µV)')
    axes['orig'].legend(ncol=2, fontsize='small')

    # 2) Compute spectrogram on channel-averaged raw signal
    raw_avg = raw.mean(axis=0)  # shape: (n_samples,)
    f, t, Sxx = spectrogram(
        raw_avg,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    # Convert to dB
    power_db = 10 * np.log10(np.maximum(Sxx, 1e-12))

    # 3) Plot time-frequency spectrogram of entire session on 'tf'
    im = axes['tf'].imshow(
        power_db,
        aspect='auto',
        origin='lower',
        extent=[t[0], t[-1], f[0], f[-1]]
    )
    axes['tf'].set_title('Time–Frequency of Entire Session (Power dB)')
    axes['tf'].set_xlabel('Time (s)')
    axes['tf'].set_ylabel('Frequency (Hz)')
    # Align x-axis of tf with orig
    orig_xlim = axes['orig'].get_xlim()
    axes['tf'].set_xlim(orig_xlim)
    axes['tf'].set_xticks(axes['orig'].get_xticks())
    cbar = fig.colorbar(im, ax=axes['tf'], pad=0.02)
    cbar.set_label('Power (dB)')

    # 4) Compute and plot PSD of raw continuous data on 'psd'
    f_psd, Pxx = welch(
        raw_avg,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    axes['psd'].plot(f_psd, Pxx)
    axes['psd'].set_title('PSD of Raw EEG')
    axes['psd'].set_xlabel('Frequency (Hz)')
    axes['psd'].set_ylabel('Power')

    plt.tight_layout()
    return fig, axes

def plotifyEEGbands(trial, band_names=None):
    band_names = band_names or ['delta', 'theta', 'alpha', 'beta', 'gamma']
    data = trial.data  # (epochs, channels, bands)
    avg_power = data.mean(axis=1)  # (epochs, bands)
    avg_data = avg_power.T         # (bands, epochs)

    fig = plt.figure(figsize=(21, 14), dpi=210, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[21, 1], height_ratios=[1, 1])

    # --- Line Plot ---
    ax_line = fig.add_subplot(gs[0, 0])
    lines = []
    for i, band in enumerate(band_names):
        line, = ax_line.plot(avg_power[:, i], label=band)
        lines.append(line)

    ax_line.set_title("Band Power Over Epochs", fontsize = 27)
    ax_line.set_xlabel("Epoch", fontsize = 21)
    ax_line.set_ylabel("Power", fontsize = 21)
    ax_line.legend(lines, band_names, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

    # --- Heatmap ---
    ax_heatmap = fig.add_subplot(gs[1, 0])
    sns.heatmap(avg_data, ax=ax_heatmap, cmap="viridis", cbar=True,
                yticklabels=band_names)
    ax_heatmap.set_title("Band × Epoch Heatmap", fontsize = 27)
    ax_heatmap.set_xlabel("Epoch", fontsize = 21)
    ax_heatmap.set_ylabel("Band", fontsize = 21)

    fig.suptitle(f"{trial.trial}", fontsize = 50)
    plt.close(fig)

    return fig