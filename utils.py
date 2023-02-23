from pathlib import Path

import soundfile as sf
from scipy import signal


def wavread(fn):
    data, sr = sf.read(fn)
    f = sf.info(fn)
    return data, sr, f.subtype


def wavwrite(fn, data, sr, subtype):
    sf.write(fn, data, sr, subtype)


def wav_convert(fn, sr_new=48000, new_subtype="PCM_16"):
    # transform any wav file to monaural signal
    assert Path(fn).suffix == ".wav"
    data, sr_org = sf.read(fn)
    if len(data) > 1:
        data = data[:, 0]
    data = signal.resample(data, int(len(data) * sr_new / sr_org))
    sf.write(fn, data, sr_new, new_subtype)
