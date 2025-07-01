import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch

from .utilities import savify

# Set consistent plot style defaults
plt.rc('figure', titlesize=33, figsize=(21, 7), dpi=210)
plt.rc('axes', titlesize=27, labelsize=21, titlepad=21)
plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)

# Define standard EEG band color palette
BAND_COLORS = {
    'delta': '#00363b',
    'theta': '#00b3ad',
    'alpha': '#b5c900',
    'beta':  '#ffa200',
    'gamma': '#FF0000'
}


def plotifyRAWdata(session, nperseg=64, noverlap=32, destination=None, show=False):
    fs = session.sampling_rate
    fig, axes = plt.subplot_mosaic(
        [['orig', 'psd'], ['tf', 'psd']],
        figsize=(21, 10), dpi=210,
        gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1.5, 1]}
    )

    raw = session.raw
    times = np.arange(raw.shape[1]) / fs
    for ch in range(raw.shape[0]):
        axes['orig'].plot(times, raw[ch], label=f'ch{ch}')
    axes['orig'].set(title='Raw Continuous EEG', xlabel='Time (s)', ylabel='Amplitude (µV)')
    axes['orig'].legend(ncol=2, fontsize='small')
    axes['orig'].title.set_y(1.01)

    raw_avg = raw.mean(axis=0)
    f, t, Sxx = spectrogram(raw_avg, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='psd')
    power_db = 10 * np.log10(np.maximum(Sxx, 1e-12))

    im = axes['tf'].imshow(power_db, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    axes['tf'].set(title='Time–Frequency (dB)', xlabel='Time (s)', ylabel='Frequency (Hz)')
    axes['tf'].set_xlim(axes['orig'].get_xlim())
    axes['tf'].set_xticks(axes['orig'].get_xticks())
    axes['tf'].title.set_y(1.01)

    fig.colorbar(im, ax=axes['tf'], pad=0.02).set_label('Power (dB)')

    f_psd, Pxx = welch(raw_avg, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density')
    axes['psd'].plot(f_psd, Pxx)
    axes['psd'].set(title='PSD of Raw EEG', xlabel='Frequency (Hz)', ylabel='Power')
    axes['psd'].title.set_y(1.01)

    plt.tight_layout()
    if destination:
        savify(fig, f"raw_summary_{session.session}.png", destination)
    if show:
        plt.show()
    return fig


def plotifyEEGbands(session, band_names=None, destination=None, show=False):
    band_names = band_names or ['delta', 'theta', 'alpha', 'beta', 'gamma']
    data = session.data
    avg_power = data.mean(axis=1)
    avg_data = avg_power.T

    fig = plt.figure(figsize=(21, 14), dpi=210, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[21, 1], height_ratios=[1, 1])

    ax_line = fig.add_subplot(gs[0, 0])
    for i, band in enumerate(band_names):
        color = BAND_COLORS.get(band, None)
        ax_line.plot(avg_power[:, i], label=band, color=color)
    ax_line.set(title="Band Power Over Epochs", xlabel="Epoch", ylabel="Power")
    ax_line.title.set_y(1.01)
    ax_line.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)

    ax_heatmap = fig.add_subplot(gs[1, 0])
    sns.heatmap(avg_data, ax=ax_heatmap, cmap="viridis", cbar=True, yticklabels=band_names)
    ax_heatmap.set(title="Band × Epoch Heatmap", xlabel="Epoch", ylabel="Band")
    ax_heatmap.title.set_y(1.01)

    fig.suptitle(f"{session.session}", fontsize=40, y=1.01)
    if destination:
        savify(fig, f"band_power_{session.session}.png", destination)
    if show:
        plt.show()
    return fig


def plot_channel_variances(data, destination=None, title="Channel Variances"):
    variances = data.var(axis=1)
    fig, ax = plt.subplots(figsize=(21, 10), dpi=210)
    ax.bar(np.arange(len(variances)), variances)
    ax.set(title=title, xlabel='Channel', ylabel='Variance')
    ax.title.set_y(1.01)
    ax.legend().set_visible(False)
    if destination:
        savify(fig, f"channel_variances.png", destination)
    plt.close(fig)
    return fig


def plot_power_spectrum(data, fs, destination=None, title="Power Spectrum"):
    f, Pxx = welch(data.mean(axis=0), fs=fs, nperseg=256)
    fig, ax = plt.subplots(figsize=(21, 10), dpi=210)
    ax.plot(f, 10 * np.log10(Pxx))
    ax.set(title=title, xlabel='Frequency (Hz)', ylabel='Power (dB)')
    ax.title.set_y(1.01)
    ax.legend().set_visible(False)
    if destination:
        savify(fig, f"power_spectrum.png", destination)
    plt.close(fig)
    return fig


def plot_filter_preview(raw, filtered, fs, channel=0, window_sec=1.0):
    """Overlay raw and filtered signals for quick inspection."""
    n = int(window_sec * fs)
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(t, raw[channel, :n], label="raw", alpha=0.7)
    ax.plot(t, filtered[channel, :n], label="filtered", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_bad_channel_removal(data, keep_indices, fs, window_sec=1.0):
    """Visualise kept and removed channels on a short time window."""
    n = int(window_sec * fs)
    t = np.arange(n) / fs
    all_idx = np.arange(data.shape[0])
    keep_mask = np.isin(all_idx, keep_indices)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    offset = np.ptp(data[:, :n]) * 0.6 or 1.0
    for ch in all_idx:
        trace = data[ch, :n]
        y = trace / (np.max(np.abs(trace)) or 1) * offset + ch * offset
        color = 'darkcyan' if keep_mask[ch] else 'red'
        ax.plot(t, y, color=color)
    ax.set_yticks(all_idx * offset)
    ax.set_yticklabels([f"Ch {i}" for i in all_idx])
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def plot_ica_topographies(ica, destination=None, title="ICA Components"):
    fig = ica.plot_components(show=False)
    fig.suptitle(title, y=1.01)
    if destination:
        savify(fig, f"ica_topographies.png", destination)
    return fig
