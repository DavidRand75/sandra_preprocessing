import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy import signal
import os
from .signal_processing import SignalProcessor, wavelet_denoising


class AudioProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.signal_processor = SignalProcessor(fs)

    def create_audio_dataframe(self, audio_file, respiration_file=None, file_path=None):
        audio = AudioSegment.from_file(audio_file)
        raw_data = audio.raw_data
        sampling_rate = audio.frame_rate
        audio_array = np.frombuffer(raw_data, dtype=np.int16)
        time_v = np.arange(len(audio_array)) / sampling_rate
        audio_data = pd.DataFrame({'time': time_v, 'raw_data': audio_array})

        audio_data['filtered'] = self.signal_processor.bandpass_filter(audio_data['raw_data'],
                                                                       lowcut=70,
                                                                       highcut=5000,
                                                                       order=6)
        audio_data['wavelet'] = wavelet_denoising(audio_data['filtered'],
                                                  wavelet='db4',
                                                  level=5,
                                                  threshold=99)
        audio_data['wavelet_smoothed'] = self.signal_processor.smoothed_signal(audio_data['wavelet'],
                                                                               window_size=50,
                                                                               iterations=4)

        if respiration_file:
            respiration_data = pd.read_csv(respiration_file)
            audio_data['Respiration'] = np.interp(audio_data.time, respiration_data['Data Set 1:Time(s)'],
                                                  respiration_data['Data Set 1:Force(N)'])

        if file_path:
            filename = audio_file.split('/')[-1].split('.')[0]
            audio_data.to_csv(os.path.join(file_path, filename + '.csv'))

        return audio_data, sampling_rate

    def respiration_shift(self, audio_df, distance=0, width=0):
        a = audio_df['wavelet_smoothed'].values
        b = audio_df['Respiration'].values
        y = signal.correlate(b, a)
        t = np.arange(-a.shape[0], a.shape[0] - 1) / self.fs
        p = signal.find_peaks(y, distance=distance, width=width)[0]
        shift = t[p[np.argmin(np.abs(t[p]))]]
        return audio_df['time'].values - shift
