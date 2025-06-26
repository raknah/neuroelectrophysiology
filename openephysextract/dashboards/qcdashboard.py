import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

from openephysextract.session import Session

class QCDashboard:
    """
    QC Dashboard for visualizing preprocessing quality metrics.

    Usage:
        dashboard = QCDashboard(sessions)
        dashboard.run()
    """
    def __init__(self, sessions):
        self.sessions = sessions
        self.app = dash.Dash(__name__)
        self._build_layout()
        self._register_callbacks()

    def _build_layout(self):
        self.app.layout = html.Div([
            html.H1("QC Dashboard", style={"textAlign": "center"}),
            html.Div([
                html.Label("Select Session:"),
                dcc.Dropdown(
                    id='session-dropdown',
                    options=[{'label': s.session, 'value': s.session} for s in self.sessions],
                    value=self.sessions[0].session
                )
            ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            html.Div([
                html.Label("View Mode:"),
                dcc.RadioItems(
                    id='mode-radio',
                    options=[
                        {'label': 'Novice', 'value': 'novice'},
                        {'label': 'Paranoid', 'value': 'paranoid'}
                    ],
                    value='novice',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                )
            ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            html.Div(id='qc-content', style={"padding": "20px"})
        ])

    def _register_callbacks(self):
        @self.app.callback(
            Output('qc-content', 'children'),
            [Input('session-dropdown', 'value'), Input('mode-radio', 'value')]
        )
        def update_content(session_id, mode):
            session = next(s for s in self.sessions if s.session == session_id)
            return self._render_session(session, mode)

    def _render_session(self, session: Session, mode: str):
        content = []
        summaries = session.qc_summaries

        # Stage: Remove Bad Channels
        if 'remove_bad' in summaries:
            stats = summaries['remove_bad']['stats']
            down = summaries['remove_bad']['downsampled']
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=down.flatten(), name='Cleaned Downsampled'))
            content.append(html.H3("Remove Bad Channels Summary"))
            content.append(dcc.Graph(figure=fig))
            content.append(html.Div(f"Channels before: {stats['n_channels_before']}, after: {stats['n_channels_after']}"))
            if mode == 'paranoid':
                full = session.load_full('remove_bad')
                fig2 = go.Figure()
                for i in range(full.shape[0]):
                    fig2.add_trace(go.Scatter(y=full[i], name=f"Ch {i}"))
                content.append(html.H4("Full Cleaned Signals"))
                content.append(dcc.Graph(figure=fig2))

        # Stage: Filtering
        if 'filter_session' in summaries:
            stats = summaries['filter_session']['stats']
            down = summaries['filter_session']['downsampled']
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=down.flatten(), name='Filtered Downsampled'))
            content.append(html.H3("Filtering Summary"))
            content.append(dcc.Graph(figure=fig))
            content.append(html.Div(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}"))
            if mode == 'paranoid':
                full = session.load_full('filter_session')
                fig2 = go.Figure()
                for i in range(full.shape[0]):
                    fig2.add_trace(go.Scatter(y=full[i], name=f"Ch {i}"))
                content.append(html.H4("Full Filtered Signals"))
                content.append(dcc.Graph(figure=fig2))

        # Stage: Downsampling
        if 'downsample_session' in summaries:
            stats = summaries['downsample_session']['stats']
            down = summaries['downsample_session']['downsampled']
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=down.flatten(), name='Downsampled'))
            content.append(html.H3("Downsampling Summary"))
            content.append(dcc.Graph(figure=fig))
            content.append(html.Div(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}"))

        # Stage: ICA
        if 'ica_sources' in session.qc_summaries or session.ica_sources is not None:
            sources = session.ica_sources
            bad = session.bad_ics or []
            fig = go.Figure()
            for i in range(sources.shape[0]):
                fig.add_trace(go.Scatter(
                    y=sources[i],
                    name=f"IC {i}",
                    line={'dash': 'dash'} if i in bad else {}
                ))
            content.append(html.H3("ICA Components"))
            content.append(html.Div(f"Rejected ICs: {bad}"))
            content.append(dcc.Graph(figure=fig))

        # Stage: Epochs
        if 'epochs' in summaries:
            stats = summaries['epochs']['stats']
            summary = summaries['epochs']['downsampled']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(summary))), y=summary, name='Epoch Mean'))
            content.append(html.H3("Epoch Summary"))
            content.append(dcc.Graph(figure=fig))
            content.append(html.Div(f"Epochs: {stats['n_epochs']}, Frame: {stats['frame']}, Stride: {stats['stride']}"))

        return content

    def run(self, host='0.0.0.0', port=8050, debug=False):
        """Launch the Dash server."""
        self.app.run_server(host=host, port=port, debug=debug)
