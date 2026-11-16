"""A region's instrumental profile must survive the copies of its fftconv1D (std::vector of regions, one copy per thread).

The spectra synthesized with a profile must equal the exact correlation of the undegraded spectra, with the padding of
fftconv1D (the last value repeated for half the profile, then the first value), for one and for several threads.
Before the fix the copy operator moved nft doubles of the nft-complex transfer function, so the upper half of it was
zero in every copy.

Run from the repository root after `python setup.py build_ext --inplace`: python -m pytest tests/test_fftconv_copy.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MilneEddington as ME


def exact(spec, psf):
    """The correlation fftconv1D computes, with its padding, in numpy."""
    n, n1 = spec.shape[-1], len(psf)
    npad = n1 + n if n1 % 2 == 0 else n1 + n - 1
    pad = np.empty(spec.shape[:-1] + (npad,))
    pad[..., :n] = spec
    pad[..., n:n + n1 // 2] = spec[..., -1:]
    pad[..., n + n1 // 2:] = spec[..., :1]
    p = np.zeros(npad)
    p[:n1] = psf / psf.sum()
    p = np.roll(p, -(n1 // 2))
    return np.fft.irfft(np.fft.rfft(pad) * np.conj(np.fft.rfft(p)), npad)[..., :n]


@pytest.mark.parametrize("nthreads", [1, 3])
def test_profile_survives_region_copies(nthreads):
    wav = 6301.9 + 0.0107 * np.arange(110)                     # Hinode's doubled grid (10.7 mA) around Fe I 6302.5
    x = 0.0107 * (np.arange(15) - 7)
    psf = np.exp(-0.5 * (x / 0.010) ** 2) * (1 + 10 * x)        # narrow and asymmetric: correlation, not convolution
    model = np.array([[[1200.0, 0.8, 0.4, 0.5, 0.03, 15.0, 0.1, 0.2, 0.8],
                       [300.0, 1.9, 2.0, -1.0, 0.025, 30.0, 0.3, 0.3, 0.7],
                       [2500.0, 0.2, 1.1, 2.0, 0.035, 8.0, 0.05, 0.1, 0.9]]])
    raw = ME.MilneEddington([[wav, None]], [6302], nthreads=nthreads).synthesize(model)
    deg = ME.MilneEddington([[wav, psf]], [6302], nthreads=nthreads).synthesize(model)
    ref = exact(raw, psf)
    scale = np.abs(raw).max(axis=-1, keepdims=True)
    assert np.abs(deg - ref).max() / scale.min() < 1e-10
