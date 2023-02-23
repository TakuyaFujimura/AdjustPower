from pathlib import Path

import numpy as np

from .utils import wavread, wavwrite


def split(filepath, n_split):
    filepath = Path(filepath)
    data, sr, subtype = wavread(filepath)
    data_len = data.shape[0]  # n_time
    split_len = data_len // n_split
    for i in range(n_split - 1):
        start = i * split_len
        end = (i + 1) * split_len
        output_path = filepath.parent / f"{filepath.stem}_i.wav"
        wavwrite(output_path, data[start:end], sr, subtype)


def concat(filepath_list, output_path):
    all_data = []
    for filepath in filepath_list:
        filepath = Path(filepath)
        data, sr, subtype = wavread(filepath)
        all_data.append(data)
    concat_data = np.concatenate(all_data, axis=0)
    wavwrite(output_path, concat_data, sr, subtype)
