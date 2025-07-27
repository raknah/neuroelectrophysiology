import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import time
import torch
import tempfile
import numpy as np
from scipy.signal import butter, filtfilt, decimate, iirnotch, detrend, sosfiltfilt
from sklearn.decomposition import FastICA

from .session import Session


class SessionStep(ABC):
    """Abstract base for preprocessing steps."""
    verbose: bool = False

    @abstractmethod
    def apply(self, session: Session, device: torch.device) -> None:
        ...

    def preferred_device(self, default: torch.device) -> torch.device:
        return default


# ------------------------
# CPU-bound steps
# ------------------------
class DetrendStep(SessionStep):
    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        mat = arr.cpu().numpy()
        mat = detrend(mat, axis=1)
        out = torch.from_numpy(np.ascontiguousarray(mat.astype(np.float32))).to(device)
        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['detrend'] = {}
        session.log_step(self)


class ASRStep(SessionStep):
    def __init__(self, cutoff: float = 3.0):
        self.cutoff = cutoff

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        X = arr.cpu().numpy()
        cov = np.cov(X)
        eigvals, eigvecs = np.linalg.eigh(cov)
        mask = eigvals <= np.median(eigvals) * self.cutoff
        proj = eigvecs[:, mask] @ eigvecs[:, mask].T
        clean = proj @ X
        out = torch.from_numpy(np.ascontiguousarray(clean.astype(np.float32))).to(device)
        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['asr'] = {'components_kept': int(mask.sum())}
        session.log_step(self)


class EOGRegressionStep(SessionStep):
    def __init__(self, eog_chs: List[int], target_chs: Optional[List[int]] = None):
        self.eog_chs = eog_chs
        self.target_chs = target_chs

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        X = arr.cpu().numpy()
        E = X[self.eog_chs].T
        T = X[self.target_chs or list(range(X.shape[0]))].T
        B, *_ = np.linalg.lstsq(E, T, rcond=None)
        clean = (T - E @ B).T.astype(np.float32)
        out = torch.from_numpy(clean).to(device)
        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['eog_regress'] = {}
        session.log_step(self)


class InterpolateStep(SessionStep):
    def __init__(self, neighbors: Dict[int, List[int]]):
        self.neighbors = neighbors

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.cpu().numpy()
        bad = set(range(arr.shape[0])) - set(getattr(session, 'good_channels', []))
        for ch in bad:
            neigh = self.neighbors.get(ch, [])
            if neigh:
                arr[ch] = arr[neigh].mean(axis=0)
        out = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(device)
        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['interpolate'] = {'bad_channels': list(bad)}
        session.log_step(self)


class FilterStep(SessionStep):
    def __init__(self, lowcut=None, highcut=None, order=4, notch_freqs=None, detrend_data=False):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freqs = notch_freqs or []
        self.detrend = detrend_data

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if not was_numpy:
            arr = arr.detach().cpu().numpy()

        arr = arr.astype(np.float64)
        fs = session.sampling_rate
        nyq = 0.5 * fs

        if self.detrend:
            arr = detrend(arr, axis=1)

        for f0 in self.notch_freqs:
            b, a = iirnotch(f0, Q=30.0, fs=fs)
            arr = filtfilt(b, a, arr, axis=1)

        if self.lowcut is not None and self.highcut is not None:
            w1 = self.lowcut / nyq
            w2 = min(self.highcut, nyq * 0.99) / nyq
            if not (0 < w1 < w2 < 1):
                raise ValueError(f"Invalid band {self.lowcut}-{self.highcut} Hz")
            sos = butter(self.order, [w1, w2], btype='band', output='sos')
            arr = sosfiltfilt(sos, arr, axis=1)

        out = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(device)
        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['filter'] = {
            'lowcut': self.lowcut,
            'highcut': self.highcut,
            'order': self.order,
            'notch_freqs': self.notch_freqs,
            'detrend': self.detrend
        }
        session.log_step(self)


class DownsampleStep(SessionStep):
    def __init__(self, target_fs: int = 300, downsample_raw: bool = True):
        self.target_fs = target_fs
        self.downsample_raw = downsample_raw

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if not was_numpy:
            arr = arr.detach().cpu().numpy()

        orig_fs = session.sampling_rate
        factor = max(1, orig_fs // self.target_fs)
        if factor > 1:
            arr = decimate(arr, factor, axis=1, ftype='fir', zero_phase=True)
            session.sampling_rate = orig_fs // factor

        arr = np.nan_to_num(arr).astype(np.float32)
        tensor = torch.from_numpy(arr).to(device)
        if self.downsample_raw:
            session.raw = arr  # store as NumPy
        session.preprocessed = tensor.cpu().numpy() if was_numpy else tensor
        session.stats['downsample'] = {'factor': factor}
        session.log_step(self)


class ICARemovalStep(SessionStep):
    def __init__(self, n_components: Optional[int] = None, reject_z: float = 3.0):
        self.n_components = n_components
        self.reject_z = reject_z

    def preferred_device(self, default):
        return torch.device('cpu')

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if not was_numpy:
            arr = arr.detach().cpu().numpy()

        mat = arr.astype(np.float32)
        ica = FastICA(n_components=self.n_components, random_state=0)
        sources = ica.fit_transform(mat.T).T
        vars_ = sources.var(axis=1)
        bad = np.where((vars_ - vars_.mean()) / vars_.std() > self.reject_z)[0]
        sources[bad] = 0
        clean = ica.inverse_transform(sources.T).T.astype(np.float32)

        session.preprocessed = clean if was_numpy else torch.from_numpy(clean).to(device)
        session.ica_sources = sources if was_numpy else torch.from_numpy(sources.astype(np.float32)).to(device)
        session.bad_ics = bad.tolist()
        session.stats['ica'] = {'removed': int(len(bad))}
        session.log_step(self)

# ------------------------
# GPU-capable steps
# ------------------------
class RemoveBadStep(SessionStep):
    def __init__(self, std=True, alpha=0.5, beta=0.5, cutoff_pct=90.0):
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.cutoff = cutoff_pct

    def apply(self, session: Session, device):
        arr = session.preprocessed if session.preprocessed is not None else session.raw
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        stds = arr.std(dim=1, keepdim=True).clamp(min=1e-6)
        means = arr.mean(dim=1, keepdim=True)
        normed = (arr - means) / stds if self.std else arr
        D = torch.cdist(normed, normed)
        D = (D - D.min()) / (D.max() - D.min()) if D.max() != D.min() else D
        C = 1 - torch.corrcoef(normed).abs()
        H = self.alpha * D + self.beta * C
        H.fill_diagonal_(1)
        scores = H.mean(dim=1)
        thresh = torch.quantile(scores, self.cutoff / 100.0)
        keep = (scores <= thresh).nonzero(as_tuple=False).squeeze()

        arr = arr[keep]
        session.preprocessed = arr.cpu().numpy() if was_numpy else arr
        session.good_channels = keep.cpu().tolist()
        session.stats['remove_bad'] = {'before': arr.size(0), 'after': keep.numel()}
        session.log_step(self)


class EpochStep(SessionStep):
    def __init__(self, frame: int, stride: int, baseline_ms: int = 0):
        self.frame = frame
        self.stride = stride
        self.baseline_ms = baseline_ms

    def apply(self, session: Session, device):
        arr = session.preprocessed
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        win = arr.unfold(dimension=1, size=self.frame, step=self.stride)
        ep = win.permute(1, 0, 2).contiguous()
        if self.baseline_ms:
            pre = int(self.baseline_ms * session.sampling_rate / 1000)
            baseline = ep[:, :, :pre].mean(dim=2, keepdim=True)
            ep = ep - baseline

        session.data = ep.cpu().numpy() if was_numpy else ep
        session.stats['epoch'] = {'n_epochs': ep.size(0)}
        session.log_step(self)


class ArtifactRejectStep(SessionStep):
    def __init__(self, threshold: float, consensus: float = 0.6):
        self.threshold = threshold
        self.consensus = consensus

    def apply(self, session: Session, device):
        arr = session.data
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        ptp = arr.amax(dim=2) - arr.amin(dim=2)
        mask = (ptp > self.threshold).float().mean(dim=1) < (1 - self.consensus)
        clean = arr[mask.bool()]
        session.data = clean.cpu().numpy() if was_numpy else clean
        session.stats['artifact_reject'] = {'kept': int(clean.size(0))}
        session.log_step(self)


class EventCompileStep(SessionStep):
    def __init__(self, event_channel: int, pre_ms: int = 10, post_ms: int = 100, baseline_correction: bool = False):
        self.event_channel = event_channel
        self.pre_ms = pre_ms
        self.post_ms = post_ms
        self.baseline_correction = baseline_correction

    def apply(self, session: Session, device):
        arr = session.preprocessed
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        ev = arr[self.event_channel - 1]
        sig = (ev >= 0.5).int()
        starts = (sig[1:] - sig[:-1]).nonzero().squeeze() + 1
        sr = session.sampling_rate
        pre = int(self.pre_ms * sr / 1000)
        post = int(self.post_ms * sr / 1000)
        span = pre + post
        trials = []
        for s in starts.tolist():
            seg = arr[:, max(0, s - pre): s + post]
            if seg.size(1) < span:
                pad = torch.zeros(arr.size(0), span, device=device)
                pad[:, :seg.size(1)] = seg
                seg = pad
            if self.baseline_correction:
                seg = seg - seg[:, :pre].mean(dim=1, keepdim=True)
            trials.append(seg)
        stack = torch.stack(trials)
        session.data = stack.cpu().numpy() if was_numpy else stack
        session.stats['event_compile'] = {'n_trials': stack.size(0)}
        session.log_step(self)


class ReReferenceStep(SessionStep):
    def __init__(self, method: str = 'average', bipolar_pairs: Optional[List[Tuple[int, int]]] = None):
        self.method = method
        self.pairs = bipolar_pairs

    def apply(self, session: Session, device):
        arr = session.preprocessed
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        if self.method == 'average':
            ref = arr.mean(dim=0, keepdim=True)
            out = arr - ref
        elif self.method == 'bipolar':
            out = torch.stack([arr[i] - arr[j] for i, j in (self.pairs or [])])
        else:
            raise ValueError(f"Unknown ref method: {self.method}")

        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['rereference'] = {}
        session.log_step(self)


class SurfaceLaplacianStep(SessionStep):
    def __init__(self, neighbors: Dict[int, List[int]]):
        self.neighbors = neighbors

    def apply(self, session: Session, device):
        arr = session.preprocessed
        was_numpy = isinstance(arr, np.ndarray)
        if was_numpy:
            arr = torch.from_numpy(arr.astype(np.float32))

        arr = arr.to(device)
        out = torch.zeros_like(arr)
        for i, neigh in self.neighbors.items():
            out[i] = arr[i] - arr[neigh].mean(dim=0) if neigh else arr[i]

        session.preprocessed = out.cpu().numpy() if was_numpy else out
        session.stats['laplacian'] = {}
        session.log_step(self)

class StandardizeStep(SessionStep):
    """Standardize data and/or preprocessed fields using z-score, min-max, robust, or logistic methods."""
    def __init__(self, method: str = 'zscore', per_epoch: bool = True):
        assert method in ('zscore', 'minmax', 'robust', 'logistic'), "method must be one of: 'zscore', 'minmax', 'robust', 'logistic'"
        self.method = method
        self.per_epoch = per_epoch

    def _standardize(self, x: torch.Tensor, dims: Tuple[int], method: str) -> torch.Tensor:
        if method == 'zscore':
            mean = x.mean(dim=dims, keepdim=True)
            std = x.std(dim=dims, keepdim=True).clamp(min=1e-6)
            return (x - mean) / std

        elif method == 'minmax':
            min_val = x.amin(dim=dims, keepdim=True)
            max_val = x.amax(dim=dims, keepdim=True)
            return (x - min_val) / (max_val - min_val).clamp(min=1e-6)

        elif method == 'robust':
            q1 = x.quantile(0.25, dim=dims, keepdim=True)
            q3 = x.quantile(0.75, dim=dims, keepdim=True)
            iqr = (q3 - q1).clamp(min=1e-6)
            return (x - q1) / iqr

        elif method == 'logistic':
            q1 = x.quantile(0.25, dim=dims, keepdim=True)
            q3 = x.quantile(0.75, dim=dims, keepdim=True)
            med = x.median(dim=dims, keepdim=True).values
            iqr = q3 - q1
            lam = (2 * np.log(3)) / iqr.clamp(min=1e-6)
            scaled = 1 / (1 + torch.exp(-lam * (x - med)))

            # fallback to minmax if IQR == 0
            min_val = x.amin(dim=dims, keepdim=True)
            max_val = x.amax(dim=dims, keepdim=True)
            fallback = (x - min_val) / (max_val - min_val).clamp(min=1e-6)
            return torch.where(iqr <= 0, fallback, scaled)

        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def apply(self, session: Session, device: torch.device) -> None:
        if session.preprocessed is not None:
            arr = session.preprocessed
            was_numpy = isinstance(arr, np.ndarray)
            if was_numpy:
                arr = torch.from_numpy(arr.astype(np.float32))
            arr = arr.to(device)
            arr = self._standardize(arr, dims=(1,), method=self.method)
            session.preprocessed = arr.cpu().numpy() if was_numpy else arr
            if self.verbose:
                print(f"[StandardizeStep] standardized preprocessed: {arr.shape} using {self.method}")

        if session.data is not None:
            arr = session.data
            was_numpy = isinstance(arr, np.ndarray)
            if was_numpy:
                arr = torch.from_numpy(arr.astype(np.float32))
            arr = arr.to(device)
            dims = (2,) if self.per_epoch else (0, 2)
            arr = self._standardize(arr, dims=dims, method=self.method)
            session.data = arr.cpu().numpy() if was_numpy else arr
            if self.verbose:
                print(f"[StandardizeStep] standardized data: {arr.shape} using {self.method}, per_epoch={self.per_epoch}")

        session.stats['standardize'] = {
            'method': self.method,
            'per_epoch': self.per_epoch,
            'preprocessed': session.preprocessed.shape if session.preprocessed is not None else None,
            'data': session.data.shape if session.data is not None else None
        }
        session.log_step(self)


class Preprocessor:
    """Runs a chain of SessionSteps with optional verbose diagnostics."""
    def __init__(self, steps: List[SessionStep], device: str = 'cpu', log: bool = False, verbose: bool = False):
        self.steps = steps
        self.device_str = device
        self.log_enabled = log
        self.verbose = verbose
        for step in self.steps:
            step.verbose = self.verbose
        self.logger = logging.getLogger('Preprocessor')
        self.logger.setLevel(logging.INFO if log else logging.WARNING)
        self.log_buffer: List[str] = []

    def _log(self, msg: str):
        if self.log_enabled:
            print(msg)
            self.log_buffer.append(msg)

    def _check_errors(self, session: Session, step: SessionStep):
        if not self.verbose:
            return
        tensor = session.preprocessed if session.preprocessed is not None else session.data
        if tensor is None:
            return
        if isinstance(tensor, torch.Tensor):
            nan_cnt = tensor.isnan().sum().item()
            inf_cnt = tensor.isinf().sum().item()
        else:
            arr = np.array(tensor)
            nan_cnt = np.isnan(arr).sum()
            inf_cnt = np.isinf(arr).sum()
        if nan_cnt:
            self._log(f"[{session.session}] ⚠️ {nan_cnt} NaNs after {step.__class__.__name__}")
        if inf_cnt:
            self._log(f"[{session.session}] ⚠️ {inf_cnt} Infs after {step.__class__.__name__}")

    def preprocess(self, sessions: List[Session], use_gpu: bool = False, downsample_factor: int = 1) -> List[Session]:
        processed: List[Session] = []
        global_dev = torch.device(self.device_str) if use_gpu else torch.device('cpu')

        for orig in tqdm(sessions, desc='Preprocessing', unit='session'):
            self._log(f"--- Session {orig.session} ---")
            start = time.perf_counter()

            new = Session(
                session=orig.session,
                experiment=orig.experiment,
                raw=orig.raw.copy(),
                preprocessed=None,
                data=None,
                sampling_rate=orig.sampling_rate,
                ch_names=orig.ch_names,
                montage=orig.montage,
                location=orig.location
            )
            new.notes = dict(orig.notes)
            new.group = orig.group
            new.events = list(orig.events) if orig.events else None
            new.history, new.stats = [], {}
            new.good_channels = None

            new.preprocessed = new.raw.astype(np.float32)
            new.data = None

            for step in self.steps:
                self._log(f"[{new.session}] → {step.__class__.__name__}")
                step_dev = step.preferred_device(global_dev)
                step.apply(new, step_dev)
                if self.verbose:
                    self._check_errors(new, step)

            if downsample_factor > 1 and isinstance(new.preprocessed, np.ndarray):
                arr = decimate(new.preprocessed, downsample_factor, axis=1, zero_phase=True)
                new.preprocessed = arr.astype(np.float32)
                new.sampling_rate //= downsample_factor

            for field in ['raw', 'preprocessed', 'data']:
                x = getattr(new, field)
                if isinstance(x, torch.Tensor):
                    setattr(new, field, x.detach().cpu().numpy())

            duration = time.perf_counter() - start
            self._log(f"[✓] Done {new.session} in {duration:.2f}s")
            processed.append(new)

        if self.log_enabled and self.log_buffer:
            f = tempfile.NamedTemporaryFile(delete=False, suffix='.log', mode='w')
            f.write("\n".join(self.log_buffer))
            f.close()
            print(f"✓ Preprocessing log at: {f.name}")

        return processed
