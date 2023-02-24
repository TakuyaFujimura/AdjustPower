import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from .utils import snr2rmsr, wavread, wavwrite


class AdjustSpeechRMS:
    def __init__(
        self,
        input_dir,
        output_dir,
        converted_dir,
        timestamp_dir,
        figure_dir,
        speech_rms_set=0.05,
        noise_snr_set=30,
        noise_threshold_ratio=0.2,
        sr=48000,
        subtype="PCM_16",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.converted_dir = Path(converted_dir)
        self.timestamp_dir = Path(timestamp_dir)
        self.figure_dir = Path(figure_dir)
        assert speech_rms_set > 0
        self.speech_rms_set = speech_rms_set
        self.noise_rms_set = speech_rms_set / snr2rmsr(noise_snr_set)
        self.noise_threshold_ratio = noise_threshold_ratio
        self.sr = sr
        self.subtype = subtype

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
                trust_repo=True,
            )
            (vad, _, read_audio, *_) = utils
            # resample a signal to 16kHz
            data16 = signal.resample(data, int(len(data) * 16000 / sr))
            timestamps = vad(data16, model, sampling_rate=16000)
            with open(self.timestamp_dir / f"{filestem}.pkl", "wb") as f:
                pickle.dump(timestamps, f)
        return timestamps

    def calc_rms(self, data, sr, speech_timestamps):
        speech_rms = np.zeros_like(data)
        noise_rms = np.zeros_like(data)
        end = 0
        for timestamp in speech_timestamps:
            start = timestamp["start"] * int(sr / 16000)
            noise_rms[end:start] = np.sqrt(np.mean(data[end:start] ** 2))
            end = timestamp["end"] * int(sr / 16000)
            speech_rms[start:end] = np.sqrt(np.mean(data[start:end] ** 2))
        noise_rms[end:] = np.sqrt(np.mean(data[end:] ** 2))
        # detect noise from speech segment using threshold
        mean_rms = np.mean(speech_rms[speech_rms > 0])
        cond1 = speech_rms < mean_rms * self.noise_threshold_ratio
        cond2 = speech_rms > 0
        noise_index = cond1 * cond2
        noise_rms[noise_index] = speech_rms[noise_index]
        speech_rms[noise_index] = 0
        return speech_rms, noise_rms, mean_rms

    def adjust_rms(self, data, sr, timestamps):
        speech_rms, noise_rms, mean_rms = self.calc_rms(data, sr, timestamps)
        index = speech_rms > 0
        data[index] *= self.speech_rms_set / speech_rms[index]
        index = noise_rms > 0
        data[index] *= self.noise_rms_set / noise_rms[index]
        return data

    def log(self, filename):
        log_path = self.output_dir / f"{Path(filename).stem}.txt"
        with open(log_path, "a") as f:
            f.write(f"==== {datetime.datetime.now()} ====\n")
            for key, value in vars(self).items():
                f.write(f"{key}: {value}\n")
            f.write("====================================\n\n")

    def __call__(self, filename_list):
        # filename_list = ["hoge.wav", "fuga.wav", ...]
        for filename in filename_list:
            self.wav_convert(filename)
            input_path = self.converted_dir / filename
            data, sr, subtype = wavread(input_path)
            assert sr == self.sr and subtype == self.subtype
            timestamps = self.get_timestamp(data, sr, filename, False)
            adjusted_data = self.adjust_rms(data, sr, timestamps)
            output_path = self.output_dir / filename
            wavwrite(output_path, adjusted_data, sr, subtype)
            self.log(filename)

    def plot_figs(
        self,
        filename,
        start_sec=0,
        end_sec=10,
        fig_size=(14, 5),
        color_list=["grey", "coral"],
        data_label_list=["Original data", "Adjusted data"],
        speech_rms_label_list=["Original speech RMS", "Adjusted speech RMS"],
        noise_rms_label_list=["Original noise RMS", "Adjusted noise RMS"],
        noise_threshold_label="noise threshold",
        save_suffix="pdf",
        label_font_size=12,
    ):
        # each color and label list is [original,  adjusted]
        assert len(color_list) == 2
        assert len(data_label_list) == 2
        assert len(speech_rms_label_list) == 2
        assert len(noise_rms_label_list) == 2

        input_path = self.converted_dir / filename
        output_path = self.output_dir / filename
        data_org, sr, subtype = wavread(input_path)
        data_adj, sr, subtype = wavread(output_path)

        # I observed the VAD result is different between using
        # the entire audio and the cropped one.
        # Therefore, we once obtain the timestamp of the entire
        # audio and crop it for visualization.
        timestamps_org = self.get_timestamp(data_org, sr, filename, load=True)
        speech_rms_org, noise_rms_org, mean_rms_org = self.calc_rms(
            data_org, sr, timestamps_org
        )
        speech_rms_adj, noise_rms_adj, _ = self.calc_rms(data_adj, sr, timestamps_org)

        # crop the segment
        data_org = data_org[int(start_sec * sr) : int(end_sec * sr)]
        data_adj = data_adj[int(start_sec * sr) : int(end_sec * sr)]
        speech_rms_org = speech_rms_org[int(start_sec * sr) : int(end_sec * sr)]
        noise_rms_org = noise_rms_org[int(start_sec * sr) : int(end_sec * sr)]
        speech_rms_adj = speech_rms_adj[int(start_sec * sr) : int(end_sec * sr)]
        noise_rms_adj = noise_rms_adj[int(start_sec * sr) : int(end_sec * sr)]

        n_org_greater = np.sum(speech_rms_org > speech_rms_adj)
        n_adj_greater = np.sum(speech_rms_adj > speech_rms_org)
        if n_org_greater >= n_adj_greater:
            data_list = [data_org, data_adj]
            speech_rms_list = [speech_rms_org, speech_rms_adj]
            noise_rms_list = [noise_rms_org, noise_rms_adj]
        else:
            data_list = [data_adj, data_org]
            speech_rms_list = [speech_rms_adj, speech_rms_org]
            noise_rms_list = [noise_rms_adj, noise_rms_org]
            color_list = color_list[::-1]
            data_label_list = data_label_list[::-1]
            speech_rms_label_list = speech_rms_label_list[::-1]
            noise_rms_label_list = noise_rms_label_list[::-1]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=fig_size)
        t = np.arange(int(start_sec * sr), int(end_sec * sr)) / sr

        # plot signals
        for i in range(2):
            axes[0].plot(t, data_list[i], color=color_list[i], label=data_label_list[i])
        ylim = np.max(np.abs(data_org)) * 2
        axes[0].set_ylim(-ylim, ylim)
        axes[0].set_ylabel("Amplitude", fontsize=label_font_size)
        axes[0].legend(loc="upper right", fontsize=label_font_size, ncol=2)

        # plot RMSs
        for i in range(2):
            axes[1].plot(
                t,
                speech_rms_list[i],
                color=color_list[i],
                label=speech_rms_label_list[i],
            )
        for i in range(2):
            axes[1].plot(
                t,
                noise_rms_list[i],
                color=color_list[i],
                linestyle="dashed",
                label=noise_rms_label_list[i],
            )
        noise_threshold = mean_rms_org * self.noise_threshold_ratio
        axes[1].axhline(
            y=noise_threshold,
            linestyle="dotted",
            color="black",
            label=noise_threshold_label,
        )

        ylim = np.max(np.abs(np.stack(speech_rms_list))) * 1.7
        axes[1].set_ylim(0, ylim)
        axes[1].set_ylabel("RMS", fontsize=label_font_size)
        axes[1].set_xlabel("Time [sec]", fontsize=label_font_size)
        axes[1].legend(loc="upper right", fontsize=label_font_size, ncol=5)

        # save the figure
        plt.tight_layout()
        plt.savefig(self.figure_dir / f"{Path(filename).stem}.{save_suffix}")

    def wav_convert(self, filename):
        # transform any wav file to monaural signal
        assert Path(filename).suffix == ".wav"
        input_path = self.input_dir / filename
        output_path = self.converted_dir / filename
        data, sr_org, subtype = wavread(input_path)
        if len(data) > 1:
            # select 0th channel
            data = data[:, 0]
        # resample
        data = signal.resample(data, int(len(data) * self.sr / sr_org))
        wavwrite(output_path, data, self.sr, self.subtype)
