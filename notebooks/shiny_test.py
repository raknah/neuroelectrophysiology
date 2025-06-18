from shiny import App, ui, render
import matplotlib.pyplot as plt

# --- UI ---
app_ui = ui.page_fluid(
    ui.output_plot("basic_plot")
)

# --- Server ---
def server(input, output, session):
    @output.plot
    def basic_plot():
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 9, 2])
        return fig

# --- App ---
app = App(app_ui, server)
