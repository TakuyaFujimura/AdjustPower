import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from util import wavread, wavwrite


class AdjustSpeechRMS:
    def __init__(
        self,
        input_dir,
        output_dir,
        timestamp_dir,
        figure_dir,
        rms_set=0.05,
        noise_threshold=0.01,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.timestamp_dir = Path(timestamp_dir)
        self.figure_dir = Path(figure_dir)
        assert noise_threshold > 0 and rms_set > 0
        self.rms_set = rms_set
        self.noise_threshold = noise_threshold

    def get_timestamp(self, data, sr, filename, load=False):
        filestem = Path(filename).stem
        if load and Path(self.timestamp_dir / f"{filestem}.pkl").exists:
            print(f"Use a saved timestamp: {filestem}.pkl")
            with open(self.timestamp_dir / f"{filestem}.pkl", "rb") as f:
                timestamps = pickle.load(f)
        else:
            print(f"Calculate a timestamp and save it as {filestem}.pkl")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
            )
            (vad, _, read_audio, *_) = utils
            # resample a signal to 16kHz
            data16 = signal.resample(data, int(len(data) * 16000 / sr))
            timestamps = vad(data16, model, sampling_rate=16000)
            with open(self.timestamp_dir / f"{filestem}.pkl", "wb") as f:
                pickle.dump(timestamps, f)
        return timestamps

    def calc_rms(self, data, sr, timestamps):
        rms = np.zeros_like(data)
        for timestamp in timestamps:
            start = timestamp["start"] * int(sr / 16000)
            end = timestamp["end"] * int(sr / 16000)
            rms[start:end] = np.sqrt(np.mean(data[start:end] ** 2))
        return rms

    def adjust_rms(self, data, sr, timestamps):
        rms_org = self.calc_rms(data, sr, timestamps)
        process_index = rms_org > self.noise_threshold
        data[process_index] *= self.rms_set / rms_org[process_index]
        return data

    def forward(self, filename_list):
        # filename_list = ["hoge.wav", "fuga.wav", ...]
        for filename in filename_list:
            input_path = self.input_dir / filename
            data, sr, subtype = wavread(input_path)
            timestamps = self.get_timestamp(data, sr, filename, False)
            adjusted_data = self.adjust_rms(data, sr, timestamps)
            output_path = self.output_dir / filename
            wavwrite(output_path, adjusted_data, sr, subtype)

    def plot_figs(
        self,
        filename,
        start_sec=0,
        end_sec=10,
        fig_size=(12, 6),
        isReverse=False,
        color_list=["grey", "coral"],
        data_label_list=["Original data", "Adjusted data"],
        rms_label_list=["Original RMS", "Adjusted RMS"],
        rms_set_label="RMS set value",
        save_suffix="pdf",
        label_font_size=14,
    ):
        # each color and label list is [original,  adjusted]
        assert start_sec > 0 and end_sec > 0
        assert len(color_list) == 2
        assert len(data_label_list) == 2
        assert len(rms_label_list) == 2

        input_path = self.input_dir / filename
        output_path = self.output_dir / filename
        original_data, sr, subtype = wavread(input_path)
        adjusted_data, sr, subtype = wavread(output_path)

        # I observed the VAD result is different between using
        # the entire audio and the cropped one.
        # Therefore, we once obtain the timestamp of the entire
        # audio and crop it for visualization.
        original_timestamps = self.get_timestamp(original_data, sr, filename, load=True)
        original_rms = self.calc_rms(original_data, sr, original_timestamps)
        adjusted_rms = self.calc_rms(adjusted_data, sr, original_timestamps)

        # crop the segment
        original_data = original_data[int(start_sec * sr) : int(end_sec * sr)]
        adjusted_data = adjusted_data[int(start_sec * sr) : int(end_sec * sr)]
        original_rms = original_rms[int(start_sec * sr) : int(end_sec * sr)]
        adjusted_rms = adjusted_rms[int(start_sec * sr) : int(end_sec * sr)]

        if isReverse:
            data_list = [adjusted_data, original_data]
            rms_list = [adjusted_rms, original_rms]
            color_list = color_list[::-1]
            data_label_list = data_label_list[::-1]
            rms_label_list = rms_label_list[::-1]
        else:
            data_list = [original_data, adjusted_data]
            rms_list = [original_rms, adjusted_rms]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=fig_size)
        t = np.arange(int(start_sec * sr), int(end_sec * sr)) / sr

        # plot signals
        for i in range(2):
            axes[0].plot(t, data_list[i], color=color_list[i], label=data_label_list[i])
        ylim = np.max(np.abs(original_data)) * 2
        axes[0].set_ylim(-ylim, ylim)
        axes[0].set_ylabel("Amplitude", fontsize=label_font_size)
        axes[0].legend(loc="upper right", fontsize=label_font_size, ncol=2)

        # plot RMSs
        for i in range(2):
            axes[1].plot(t, rms_list[i], color=color_list[i], label=rms_label_list[i])
        axes[1].axhline(
            y=self.rms_set, linestyle="dashed", color="black", label=rms_set_label
        )
        ylim = self.rms_set * 2.4
        axes[1].set_ylim(-ylim, ylim)
        axes[1].set_ylabel("RMS", fontsize=label_font_size)
        axes[1].set_xlabel("Time [sec]", fontsize=label_font_size)
        axes[1].legend(loc="upper right", fontsize=label_font_size, ncol=2)

        # save the figure
        plt.tight_layout()
        plt.savefig(self.figure_dir / f"{filename}.{save_suffix}")
