import numpy as np

from scipy.signal.windows import dpss
from scipy.fft import rfftfreq, rfft
from .trial import Trial

def bandpower(trial, bands=None, nw=2):
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 25),
            'gamma': (25, 50)
        }

    fs = trial.sampling_rate
    epochs, channels, samples = trial.data.shape
    frequencies = rfftfreq(samples, d=1/fs)

    tapers = dpss(samples, NW=nw, Kmax=2*nw - 1)
    k = tapers.shape[0]

    psd = np.zeros((epochs, channels, len(frequencies)))
    features = np.zeros((epochs, channels, len(bands)))

    for epoch in range(epochs):
        for channel in range(channels):
            signal = trial.data[epoch, channel, :]
            spectrum_sum = np.zeros_like(frequencies)

            for taper in tapers:
                tapered = signal * taper
                spectrum = np.abs(rfft(tapered)) ** 2
                spectrum_sum += spectrum

            psd[epoch, channel] = spectrum_sum / k

            for idx, (band_name, (low, high)) in enumerate(bands.items()):
                mask = (frequencies >= low) & (frequencies <= high)
                features[epoch, channel, idx] = np.sum(psd[epoch, channel, mask])

    trial.data = features  # shape: (epochs, channels, bands)
    return trial

def logistic_scaler(trial):

    features = trial.data
    scaled = np.zeros_like(features)

    for i in range(trial.data.shape[1]):
        x = features[:, i]
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        median = np.median(x)
        IQR = q3 - q1

        if IQR == 0:
            scaled[:, i] = (x - np.min(x))/(np.max(x) - np.min(x))
        else:
            lam = (2 * np.log(3)) / IQR
            scaled[:, i] = 1 / (1 + np.exp(-lam * (x - median)))
    trial.data = scaled

    return trial