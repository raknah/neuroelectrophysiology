import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, '.')
from openephysextract.viewer import Viewer

def test_plot_eeg_runs():
    n_channels = 2
    n_samples = 3300
    n_events = 3
    data = np.random.randn(n_channels, n_samples, n_events)
    extracted = [{'data': data, 'notes': {'currentLevel': [1]*n_events, 'session': 'demo'}}]
    viewer = Viewer(extracted, sampling_rate=1000)
    fig, delays = viewer.plot_eeg(0)
    assert 'positive peaks' in delays and 'negative peaks' in delays
