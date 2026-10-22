"""H5 subset of DasPrep.py used by this pipeline."""

from pathlib import Path

import h5py
import numpy as np
from scipy import signal


def read_das(filename, *, metadata=False, ch1=0, ch2=None):
    """Read the Silixa-style H5 layout used by Highway_2024_0510."""
    path = Path(filename)
    if path.suffix.lower() != ".h5":
        raise ValueError(f"Only H5 input is supported by the main pipeline: {path}")

    with h5py.File(path, "r") as h5_file:
        acquisition = h5_file["Acquisition"]
        nch = int(acquisition.attrs["NumberOfLoci"])
        if metadata:
            time_array = h5_file["Acquisition/Raw[0]/RawDataTime/"]
            return {
                "dt": float(np.diff(time_array).mean() / 1e6),
                "nt": len(time_array),
                "dx": float(acquisition.attrs["SpatialSamplingInterval"]),
                "nch": nch,
                "GL": float(acquisition.attrs["GaugeLength"]),
                "headers": dict(acquisition.attrs),
            }

        end = nch if ch2 is None else ch2
        dataset = h5_file["Acquisition/Raw[0]/RawData/"]
        if dataset.shape[0] == nch:
            return dataset[ch1:end, :]
        return dataset[:, ch1:end].T


def das_preprocess(data):
    output = signal.detrend(data)
    return output - np.median(output, axis=0)


def tapering(data, alpha):
    window = signal.windows.tukey(data.shape[1], alpha)
    return data * window[None, :]


def lowpass(data, dt, fh):
    sos = signal.butter(6, fh, "lp", fs=1 / dt, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)
