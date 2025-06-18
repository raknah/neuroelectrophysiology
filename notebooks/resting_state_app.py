from shiny import App, ui, render
import matplotlib.pyplot as plt
from openephysextract.utilities import loadify
from openephysextract.plot import plotifyEEGbands  # <- make sure this returns fig
import os

# --- Load Data ---
DATA_PATH = "/Users/fomo/Documents/Research/UNIC Research/Motor Evoked Potentials/resting state analysis"
processed = loadify(DATA_PATH, "logistic-scaled")  # <- must exist
trial_names = [f"Trial {i}" for i in range(len(processed))]

# --- UI ---
app_ui = ui.page_fluid(
    ui.input_select("trial", "Select Trial", choices=trial_names),
    ui.output_plot("band_plot")  # <- DO NOT overwrite the name 'output'
)

# --- Server ---
def server(input, output, session):
    @output
    @render.plot
    def band_plot():
        idx = trial_names.index(input.trial())
        trial = processed[idx]
        return plotifyEEGbands(trial)

# --- App ---
app = App(app_ui, server)
