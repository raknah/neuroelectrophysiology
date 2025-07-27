import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
from scipy.signal import spectrogram, welch, butter, filtfilt
import base64, pickle, io
from openephysextract.session import Session


# Preserve original band colors

BAND_COLORS = {
    'delta': '#00363b',
    'theta': '#00b3ad',
    'alpha': '#b5c900',
    'beta':  '#ffa200',
    'gamma': '#FF0000'
}

# Initialize Dash app
def serve_layout():
    return html.Div([
        html.H1("EEG Session Explorer", style={'textAlign':'center'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag & Drop or ', html.A('Select Session File')]),
            style={
                'width': '100%', 'height': '60px',
                'lineHeight': '60px', 'borderWidth': '1px',
                'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='session-info'),
        html.Div(id='channel-selector'),
        dcc.Dropdown(
            id='plot-type',
            options=[
                {'label': 'Raw Continuous EEG', 'value': 'raw'},
                {'label': 'Time–Frequency',     'value': 'tf'},
                {'label': 'PSD',                'value': 'psd'},
                {'label': 'Band Power',         'value': 'band'},
                {'label': 'Channel Variances',  'value': 'var'},
                {'label': 'Filter Preview',     'value': 'filter'},
                {'label': 'Bad Channel Removal','value': 'bad'}
            ],
            placeholder='Select Plot',
            style={'width': '50%', 'margin': '10px'}
        ),
        dcc.Graph(id='eeg-graph'),
        dcc.Store(id='session-store')
    ])

app = dash.Dash(__name__)
app.layout = serve_layout
server = app.server

# Helper: decode upload contents
def parse_session(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.endswith('.pkl'):
        return pickle.load(io.BytesIO(decoded))
    else:
        arr = np.load(io.BytesIO(decoded))
        return Session(session=filename, experiment='app', raw=arr)

# Store session and update UI
@app.callback(
    [Output('session-store', 'data'),
     Output('session-info', 'children'),
     Output('channel-selector', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_session(contents, fname):
    if contents is None:
        return None, html.Div(), html.Div()
    sess = parse_session(contents, fname)
    info = html.Div([
        html.P(f"Session ID: {sess.session}"),
        html.P(f"Experiment: {sess.experiment}"),
        html.P(f"Sampling Rate: {sess.sampling_rate} Hz")
    ])
    # Channel checklist for bad channel removal
    n_ch = sess.raw.shape[0]
    checklist = dcc.Checklist(
        id='keep-channels',
        options=[{'label': f'Ch {i}', 'value': i} for i in range(n_ch)],
        value=list(range(n_ch)),
        labelStyle={'display': 'inline-block'}
    )
    return {'contents': contents, 'filename': fname}, info, html.Div([html.Label('Channels to Keep'), checklist])

# Generate plots
@app.callback(
    Output('eeg-graph', 'figure'),
    [Input('plot-type', 'value')],
    [State('session-store', 'data'), State('keep-channels', 'value')]
)
def update_graph(plot_type, session_data, keep):
    if not session_data or not plot_type:
        return go.Figure()
    sess = parse_session(session_data['contents'], session_data['filename'])
    raw = sess.raw.cpu().numpy() if hasattr(sess.raw, 'cpu') else sess.raw
    data = sess.data.cpu().numpy() if sess.data is not None else (sess.preprocessed.cpu().numpy() if sess.preprocessed is not None else raw)
    fs = sess.sampling_rate

    if plot_type == 'raw':
        times = np.arange(raw.shape[1]) / fs
        fig = go.Figure([
            go.Scatter(x=times, y=raw[ch], mode='lines', name=f'ch{ch}', line={'width':1, 'opacity':0.7})
            for ch in range(raw.shape[0])
        ])
        fig.update_layout(title='Raw Continuous EEG', xaxis_title='Time (s)', yaxis_title='Amplitude')

    elif plot_type == 'tf':
        avg = raw.mean(axis=0)
        f, t, S = spectrogram(avg, fs, nperseg=128, noverlap=64)
        power = 10 * np.log10(np.maximum(S, 1e-12))
        fig = go.Figure(
            go.Heatmap(z=power, x=t, y=f, colorscale='Viridis')
        )
        fig.update_layout(title='Time–Frequency (dB)', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')

    elif plot_type == 'psd':
        avg = raw.mean(axis=0)
        f, Pxx = welch(avg, fs, nperseg=256)
        fig = go.Figure(go.Scatter(x=f, y=Pxx, mode='lines'))
        fig.update_layout(title='Power Spectral Density', xaxis_title='Frequency (Hz)', yaxis_title='Power')

    elif plot_type == 'band':
        band_names = list(BAND_COLORS.keys())
        # assume data shape (epochs, bands)
        epochs = np.arange(data.shape[0])
        fig = go.Figure()
        for i, band in enumerate(band_names):
            if data.ndim == 2 and data.shape[1] == len(band_names):
                y = data[:, i]
            else:
                y = data.mean(axis=1)
            fig.add_trace(go.Scatter(x=epochs, y=y, mode='lines', name=band, line={'color':BAND_COLORS[band]}))
        fig.update_layout(title='Band Power Over Epochs', xaxis_title='Epoch', yaxis_title='Power')

    elif plot_type == 'var':
        var = data.var(axis=1)
        fig = go.Figure(go.Bar(x=np.arange(len(var)), y=var))
        fig.update_layout(title='Channel Variances', xaxis_title='Channel', yaxis_title='Variance')

    elif plot_type == 'filter':
        b, a = butter(4, [8/(fs/2), 12/(fs/2)], btype='band')
        filt = filtfilt(b, a, raw[0])
        n = int(fs)
        t = np.arange(n) / fs
        fig = go.Figure([
            go.Scatter(x=t, y=raw[0, :n], mode='lines', name='raw', line={'opacity':0.7}),
            go.Scatter(x=t, y=filt[:n], mode='lines', name='filtered', line={'opacity':0.7})
        ])
        fig.update_layout(title='Filter Preview (Ch 0)', xaxis_title='Time (s)', yaxis_title='Amplitude')

    else:  # 'bad'
        n = int(fs)
        t = np.arange(n) / fs
        fig = go.Figure()
        offset = np.ptp(raw[:, :n]) * 0.5 or 1.0
        for i in range(raw.shape[0]):
            y = raw[i, :n] / (np.max(np.abs(raw[i, :n])) or 1) * offset + i*offset
            color = 'darkcyan' if i in keep else 'red'
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=f'ch{i}', line={'color': color}))
        fig.update_layout(title='Bad Channel Removal Preview', xaxis_title='Time (s)', yaxis_ticktext=[f'Ch {i}' for i in range(raw.shape[0])])

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
