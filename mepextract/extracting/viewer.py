import numpy as np
import pandas as pd
import ipywidgets as widgets

from scipy.ndimage import gaussian_filter1d as gf
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from IPython.display import display

def find_meps(data, search, threshold, time_in_ms, ax):

    delays = {'positive peaks': [], 'negative peaks': []}

    pos_peaks, pos_parameters = find_peaks(
        data,
        height = search['height'],
        distance = search['distance'],
        width = search['width'])

    neg_peaks, neg_parameters = find_peaks(-data, height = search['height'], distance = search['distance'], width = search['width'])

    for peak in pos_peaks:
        if 330 < peak and data[peak] > threshold:
            ax.plot(time_in_ms[peak], data[peak], 'x', color='red')
            amplitude = data[peak]
            delays['positive peaks'].append((peak, amplitude))

    for peak in neg_peaks:
        if 330 < peak < 500 and data[peak] < -0.1:
            ax.plot(time_in_ms[peak], data[peak], 'x', color='blue')
            amplitude = data[peak]
            delays['negative peaks'].append((peak, np.abs(amplitude)))

    return delays

class Viewer:
    def __init__(self, extracted, sampling_rate):
        self.extracted = extracted
        self.sampling_rate = sampling_rate

        # useful variables
        self.count = len(extracted)
        self.conversion_to_ms = 1/self.sampling_rate * 1000
        self.time_in_ms = np.arange(-300, 3000) * (1/self.sampling_rate * 1000)
        self.last_delays = None

        # accepted trials
        self.accepted_indices = set()
        self.accepted = None

        # outputs
        self.spreadsheet = None

    def extract(self):
        self.accepted = [self.extracted[index] for index, _ in enumerate(self.extracted) if index in self.accepted_indices]
        self.spreadsheet = pd.DataFrame([trial['notes'] for index, trial in enumerate(self.extracted) if index in self.accepted_indices])
        return self.accepted, self.spreadsheet

    def reset_accept(self):
        self.accepted_indices = set()
        self.accepted = None
        self.spreadsheet = None
        return print("Accepted Set Reset.")

    def plot_trial(self, trial_index, window, std=False, log=False, peak=False,
                   threshold=None, height=None, distance=None, width=None):

        search = {'height': height, 'distance': distance, 'width': width}

        fig, ax = plt.subplots(1, 1, figsize=(21, 10), dpi=210)

        temp = self.extracted[trial_index]['data']
        notes = self.extracted[trial_index]['notes']
        current = notes['currentLevel']
        title = notes['session']
        group = int(notes['phenoCode'])

        mean_arrays = []
        std_arrays = []
        delays = {'positive peaks': [], 'negative peaks': []}

        for channel in range(temp.shape[0]):

            # current correction
            data = temp[channel, :, :]
            current_matrix = np.array(current)[np.newaxis, np.newaxis]
            data = data/current_matrix

            # baseline correction
            baseline = temp[channel, :int(10/self.conversion_to_ms), :].mean(axis=1).mean(axis=0)
            data = data - baseline

            # event mean and std
            mean_events = (data.mean(axis=1)).flatten()
            std_events = (data.std(axis=1)).flatten()

            # smooth signal
            smoothed_mean = gf(mean_events, sigma=1.25)
            smoothed_std = gf(std_events, sigma=1.25)

            standardised = (smoothed_mean - smoothed_mean.mean())/smoothed_mean.std()

            # standardised scale
            if std:
                ax.plot(self.time_in_ms, standardised, label=f'Channel {channel}')
                ax.fill_between(self.time_in_ms, standardised - 1, standardised + 1, alpha=0.5)
                mean_arrays.append(standardised)
                std_arrays.append(np.ones_like(standardised))
                if peak:
                    delays['positive peaks'].append(find_meps(standardised, search, threshold, self.time_in_ms, ax)['positive peaks'])
                    delays['negative peaks'].append(find_meps(standardised, search, threshold, self.time_in_ms, ax)['negative peaks'])

            else:
                ax.plot(self.time_in_ms, smoothed_mean, label=f'Channel {channel}')
                ax.fill_between(self.time_in_ms, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.5)
                mean_arrays.append(smoothed_mean)
                std_arrays.append(smoothed_std)
                if peak:
                    delays['positive peaks'].append(find_meps(smoothed_mean, search, threshold, self.time_in_ms, ax)['positive peaks'])
                    delays['negative peaks'].append(find_meps(smoothed_mean, search, threshold, self.time_in_ms, ax)['negative peaks'])

        stacked_mean = np.vstack(mean_arrays)
        stacked_std = np.vstack(std_arrays)

        self.extracted[trial_index]["event_mean"] = stacked_mean
        self.extracted[trial_index]["event_std"] = stacked_std
        self.last_delays = delays

        ax.set_title(f'Trial {title}', fontsize=27)
        ax.set_xlabel('Time (ms)', fontsize=21)

        if not log:
            ax.set_xlim(window, window+12)

        if log:
            ax.set_xlabel('Log-Time', fontsize=21)
            ax.set_xscale('log')

        if std:
            ax.set_ylabel('Standardised Amplitude', fontsize=21)
        else:
            ax.set_ylabel('Amplitude (uV)', fontsize=21)

        plt.tick_params(axis='both', which='major', labelsize=21)  # Major tick labels (numbers)
        plt.tick_params(axis='both', which='minor', labelsize=21)

        ax.legend(fontsize=21)

        plt.tight_layout()

        return fig, delays

    def accept_view(self):

        print_output = widgets.Output()

        # trial counter
        index_slider = widgets.IntText(value=0, min=0, max=self.count-1, description="Trial:", layout=widgets.Layout(width='121px'))

        increment_button = widgets.Button(description="+", layout=widgets.Layout(width='75%'))
        decrement_button = widgets.Button(description="-", layout=widgets.Layout(width='75%'))

        def increment(change):
            index_slider.value += 1

        def decrement(change):
            index_slider.value -= 1

        increment_button.on_click(increment)
        decrement_button.on_click(decrement)

        # window slider
        window_slider = widgets.BoundedIntText(
            value=-2,
            min=-10,
            max=90,
            step=1,
            description='Start',
            layout=widgets.Layout(width='150px')
        )
        sliders = widgets.HBox([index_slider, decrement_button, increment_button, window_slider])

        # accept/reject buttons
        accept_button = widgets.Button(description="Accept", button_style='success')
        reject_button = widgets.Button(description="Reject", button_style='danger')
        accept_reject = widgets.HBox([accept_button, reject_button])

        # plotting options
        std_checkbox = widgets.Checkbox(value=False, description='Standardised')
        log_checkbox = widgets.Checkbox(value=False, description='Log')
        plot_checkboxes = widgets.HBox([std_checkbox, log_checkbox])

        # peak finding options
        peak_checkbox = widgets.Checkbox(value=False, description='Find Peaks')
        height_input = widgets.FloatText(value=0.1, description='Height', layout=widgets.Layout(width='150px'))
        distance_input = widgets.IntText(value=10, description='Distance', layout=widgets.Layout(width='150px'))
        width_input = widgets.IntText(value=10, description='Width', layout=widgets.Layout(width='150px'))
        threshold_input = widgets.FloatText(value=0.1, description='Threshold', layout=widgets.Layout(width='150px'))
        save_button = widgets.Button(description="Save Peaks", button_style='info', layout=widgets.Layout(width='210px'))

        search_input = widgets.HBox([height_input, distance_input, width_input, threshold_input])

        def accept(b):
            current_index = index_slider.value
            with print_output:
                if current_index not in self.accepted_indices:
                    self.accepted_indices.add(current_index)
                    print(f"Trial {current_index} accepted")
                else:
                    print(f"Trial {current_index} is already accepted")
            index_slider.value += 1

        def reject(b):
            current_index = index_slider.value
            with print_output:
                print(f"Trial {current_index} rejected")
            index_slider.value += 1

        def save_peaks(b):
            current_index = index_slider.value
            name = self.extracted[current_index]['trial']
            with print_output:
                if self.last_delays is not None:
                    self.extracted[current_index]['positive peaks'] = self.last_delays['positive peaks']
                    self.extracted[current_index]['negative peaks'] = self.last_delays['negative peaks']
                    print(f"Peaks saved for index: {current_index} session: {name}\n- Positive Peaks: {self.last_delays['positive peaks']} \n- Negative Peaks: {self.last_delays['negative peaks']}")
                else:
                    print("No peaks to save")

        accept_button.on_click(accept)
        reject_button.on_click(reject)
        save_button.on_click(save_peaks)

        output = widgets.interactive_output(self.plot_trial, {
            'trial_index': index_slider,
            'window': window_slider,
            'std': std_checkbox,
            'log': log_checkbox,
            'peak': peak_checkbox,
            'threshold': threshold_input,
            'height': height_input,
            'distance': distance_input,
            'width': width_input
        })

        output.layout = widgets.Layout(width='95%', height='1000')

        # arrange widgets horizontally
        widgets_top = widgets.HBox([sliders, accept_reject, plot_checkboxes], layout=widgets.Layout(align_items='center'))
        widgets_bottom = widgets.HBox([peak_checkbox, search_input, save_button], layout=widgets.Layout(align_items='center'))

        spacer = widgets.HTML("<br>")

        # display the layout
        display(widgets_top, spacer, output, spacer, widgets_bottom)

        display(print_output)