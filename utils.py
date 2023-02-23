from pathlib import Path

import numpy as np
import soundfile as sf


def wavread(fn):
    data, sr = sf.read(fn)
    f = sf.info(fn)
    return data, sr, f.subtype


def wavwrite(fn, data, sr, subtype):
    sf.write(fn, data, sr, subtype)


def wav_split(filepath, n_split):
    filepath = Path(filepath)
    data, sr, subtype = wavread(filepath)
    data_len = data.shape[0]  # n_time
    split_len = data_len // n_split
    for i in range(n_split - 1):
        start = i * split_len
        end = (i + 1) * split_len
        output_path = filepath.parent / f"{filepath.stem}_{i}.wav"
        wavwrite(output_path, data[start:end], sr, subtype)
    output_path = filepath.parent / f"{filepath.stem}_{i+1}.wav"
    wavwrite(output_path, data[end:], sr, subtype)


def wav_concat(filepath_list, output_path):
    all_data = []
    for filepath in filepath_list:
        filepath = Path(filepath)
        data, sr, subtype = wavread(filepath)
        all_data.append(data)
    concat_data = np.concatenate(all_data, axis=0)
    wavwrite(output_path, concat_data, sr, subtype)


def rmsr2snr(rsm_ratio):
    # rsm_ratio = s_rsm / n_rsm
    power_ratio = rsm_ratio**2
    return 10 * np.log10(power_ratio)


def snr2rmsr(snr):
    power_ratio = 10 ** (snr / 10)
    return np.sqrt(power_ratio)
