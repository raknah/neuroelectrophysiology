import numpy as np
import pandas as pd
import ipywidgets as widgets
from docutils.nodes import description

from scipy.ndimage import gaussian_filter1d as gf
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from IPython.display import display

def find_meps(data, search, threshold, time_in_ms, ax, pre_stimulus, cutoff):
    """
    Find MEPs in the data using the search parameters provided.
    :param pre_stimulus:
    :param data:
    :param search:
    :param threshold:
    :param time_in_ms:
    :param ax:
    :return:
    """

    delays = {'positive peaks': [], 'negative peaks': []}
    def get_peaks(data, invert=False):
        post_artefact = data[pre_stimulus:]
        peak_data = -post_artefact if invert else post_artefact
        return find_peaks(
            peak_data,
            height = search.get('height'),
            threshold = search.get('threshold'),
            prominence = search.get('prominence'),
            distance = search.get('distance'),
            width = search.get('width'),
            plateau_size = search.get('plateau_size'),
        )

    # extract positive and negative peaks
    pos_peaks, _ = get_peaks(data)
    neg_peaks, _ = get_peaks(data, invert=True)

    cutoff = len(data) if cutoff is None else cutoff

    for peak in pos_peaks:
        peak += pre_stimulus
        if data[peak] > threshold and peak < cutoff:
            ax.plot(time_in_ms[peak], data[peak], 'x', color='red')
            amplitude = data[peak]
            delays['positive peaks'].append((peak, amplitude))

    for peak in neg_peaks:
        peak += pre_stimulus
        if data[peak] < -threshold and peak < cutoff:
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

        # classification
        self.accepted_indices = set()
        self.unsure_indices = set()
        self.rejected_indices = set()

        # classified data
        self.accepted = None
        self.unsure = None
        self.rejected = None

    def extract(self):
        self.accepted = [self.extracted[index] for index, _ in enumerate(self.extracted) if index in self.accepted_indices]
        self.unsure = [self.extracted[index] for index, _ in enumerate(self.extracted) if index in self.unsure_indices]
        self.rejected = [self.extracted[index] for index, _ in enumerate(self.extracted) if index in self.rejected_indices]

        return self.accepted, self.unsure, self.rejected

    def reset(self):
        self.accepted_indices = set()
        self.accepted = None

        self.unsure_indices = set()
        self.unsure = None

        self.rejected_indices = set()
        self.rejected = None
        return print("Accepted Set Reset.")

    def plot_trial(self, trial_index, window_start = -2, window_end = 10, std=False, x_log=False, y_log=False, peak=False,
                   threshold=None, height=None, distance=None, width=None, prominence=None, pre_stimulus=None, cutoff=None):

        search = {'height': height, 'distance': distance, 'width': width, 'prominence': prominence}

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
                    for peak in find_meps(standardised, search, threshold, self.time_in_ms, ax, pre_stimulus, cutoff)['positive peaks']:
                        delay, amplitude = peak
                        amplitude = smoothed_mean[delay]
                        delays['positive peaks'].append((delay, amplitude))
                    for peak in find_meps(standardised, search, threshold, self.time_in_ms, ax, pre_stimulus, cutoff)['negative peaks']:
                        delay, amplitude = peak
                        amplitude = smoothed_mean[delay]
                        delays['negative peaks'].append((delay, amplitude))
            else:
                ax.plot(self.time_in_ms, smoothed_mean, label=f'Channel {channel}')
                ax.fill_between(self.time_in_ms, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.5)
                mean_arrays.append(smoothed_mean)
                std_arrays.append(smoothed_std)
                if peak:
                    for peak in find_meps(smoothed_mean, search, threshold, self.time_in_ms, ax, pre_stimulus, cutoff)['positive peaks']:
                        delays['positive peaks'].append(peak)
                    for peak in find_meps(smoothed_mean, search, threshold, self.time_in_ms, ax, pre_stimulus, cutoff)['negative peaks']:
                        delays['negative peaks'].append(peak)

        stacked_mean = np.vstack(mean_arrays)
        stacked_std = np.vstack(std_arrays)

        self.extracted[trial_index]["event_mean"] = stacked_mean
        self.extracted[trial_index]["event_std"] = stacked_std
        self.last_delays = delays

        ax.set_title(f'Trial {title}', fontsize=27)
        ax.set_xlabel('Time (ms)', fontsize=21)

        if not x_log:
            ax.set_xlim(window_start, window_end)

        if x_log:
            ax.set_xlabel('Log-Time', fontsize=21)
            ax.set_xscale('log')

        if y_log:
            ax.set_ylabel('Log-Amplitude', fontsize = 21)
            ax.set_yscale('log')

        if std:
            ax.set_ylabel('Standardised Amplitude', fontsize=21)
        else:
            ax.set_ylabel('Amplitude (uV)', fontsize=21)

        plt.tick_params(axis='both', which='major', labelsize=21)  # Major tick labels (numbers)
        plt.tick_params(axis='both', which='minor', labelsize=21)

        ax.legend(fontsize=21)

        plt.tight_layout()

        return fig, delays

    def classifier(self):

        print_output = widgets.Output()

        # trial counter
        index_slider = widgets.IntText(
            value=0,
            min=0,
            max=self.count-1,
            description="Trial:",
            layout=widgets.Layout(width='150px')
        )

        # window slider
        window_start = widgets.BoundedIntText(
            value=-2,
            min=-10,
            max=90,
            step=1,
            description='Start:',
            layout=widgets.Layout(width='150px')
        )

        window_end = widgets.BoundedIntText(
            value = 10,
            min = 0,
            max = 100,
            step = 1,
            description = 'End:',
            layout=widgets.Layout(width = '150px')
        )

        # accept/reject buttons
        accept_button = widgets.Button(description="Accept", button_style='success')
        unsure_button = widgets.Button(description="Review", button_style='warning')
        reject_button = widgets.Button(description="Reject", button_style='danger')


        # plotting options
        std_checkbox = widgets.Checkbox(value=False, description='Std')
        x_log_checkbox = widgets.Checkbox(value=False, description='x-log')
        y_log_checkbox = widgets.Checkbox(value=False, description = 'y-log')


        # peak finding options
        peak_checkbox = widgets.Checkbox(value=False, description='Find Peaks')
        pre_stimulus_samples = widgets.IntText(value=330, description='From', layout=widgets.Layout(width='140px'))
        height_input = widgets.FloatText(value=1, description='Height', layout=widgets.Layout(width='140px'))
        distance_input = widgets.IntText(value=120, description='Distance', layout=widgets.Layout(width='140px'))
        width_input = widgets.IntText(value=3, description='Width', layout=widgets.Layout(width='140px'))
        prominence_input = widgets.FloatText(value=1, description='Prominence', layout=widgets.Layout(width='140px'))
        threshold_input = widgets.FloatText(value=0.5, description='Threshold', layout=widgets.Layout(width='140px'))
        cutoff_input = widgets.IntText(value=None, description='To', layout=widgets.Layout(width='140px'))
        save_button = widgets.Button(description="Save Peaks", button_style='info', layout=widgets.Layout(width='140px'))

        search_input = widgets.HBox(
            [pre_stimulus_samples, cutoff_input, height_input, distance_input, width_input, prominence_input, threshold_input],
            layout=widgets.Layout(margin='0px')
        )

        def accept(b):
            current_index = index_slider.value
            with print_output:
                if current_index in self.rejected_indices:
                    self.rejected_indices.remove(current_index)
                if current_index in self.unsure_indices:
                    self.unsure_indices.remove(current_index)
                if current_index not in self.accepted_indices:
                    self.accepted_indices.add(current_index)
                    print(f"ACCEPTED Trial {current_index}")
                else:
                    print(f"ACCEPTED Trial {current_index} is already accepted")
            index_slider.value += 1

        def reject(b):
            current_index = index_slider.value
            with print_output:
                if current_index in self.accepted_indices:
                    self.accepted_indices.remove(current_index)
                if current_index in self.unsure_indices:
                    self.unsure_indices.remove(current_index)
                if current_index not in self.rejected_indices:
                    self.rejected_indices.add(current_index)
                    print(f"REJECTED Trial {current_index} rejected")
                else:
                    print(f"REJECTED Trial {current_index} is already rejected")
            index_slider.value += 1

        def unsure(b):
            current_index = index_slider.value
            with print_output:
                if current_index in self.accepted_indices:
                    self.accepted_indices.remove(current_index)
                if current_index in self.rejected_indices:
                    self.rejected_indices.remove(current_index)
                if current_index not in self.unsure_indices:
                    self.unsure_indices.add(current_index)
                    print(f"REVIEW Trial {current_index}")
                else:
                    print(f"REVIEW Trial {current_index} is already marked for review")
            index_slider.value += 1

        def save_peaks(b):
            current_index = index_slider.value
            name = self.extracted[current_index]['trial']
            with print_output:
                if self.last_delays is not None:
                    self.extracted[current_index]['positive peaks'] = self.last_delays['positive peaks']
                    self.extracted[current_index]['negative peaks'] = self.last_delays['negative peaks']
                else:
                    print("No peaks to save")
                if current_index in self.rejected_indices:
                    self.rejected_indices.remove(current_index)
                if current_index in self.unsure_indices:
                    self.unsure_indices.remove(current_index)
                if current_index not in self.accepted_indices:
                    self.accepted_indices.add(current_index)
                    print(f"ACCEPTED Trial {current_index} ({name}) Positive: {self.last_delays['positive peaks']}, Negative: {self.last_delays['negative peaks']}")
                else:
                    print(f"ACCEPTED Trial {current_index} is already accepted")
            index_slider.value += 1

        accept_button.on_click(accept)
        unsure_button.on_click(unsure)
        reject_button.on_click(reject)
        save_button.on_click(save_peaks)

        output = widgets.interactive_output(self.plot_trial, {
            'trial_index': index_slider,
            'window_start': window_start,
            'window_end': window_end,
            'std': std_checkbox,
            'x_log': x_log_checkbox,
            'y_log': y_log_checkbox,
            'peak': peak_checkbox,
            'threshold': threshold_input,
            'height': height_input,
            'distance': distance_input,
            'width': width_input,
            'prominence': prominence_input,
            'pre_stimulus': pre_stimulus_samples,
            'cutoff': cutoff_input
        })

        output.layout = widgets.Layout(width='80%')

        # arrange widgets horizontally
        widgets_top = widgets.HBox(
            [index_slider, window_start, window_end, accept_button, unsure_button, reject_button, std_checkbox, x_log_checkbox, y_log_checkbox],
            layout=widgets.Layout(width='75%', justify_content='space-between')
        )


        widgets_bottom = widgets.HBox(
            [peak_checkbox, search_input, save_button],
            layout=widgets.Layout(width='75%', align_content='center')
        )

        spacer = widgets.HTML("<br>")

        # display the layout
        display(print_output)
        display(widgets_top, spacer, output, spacer, widgets_bottom)

