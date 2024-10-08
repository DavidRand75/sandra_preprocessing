import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy import signal
import librosa
import os
from signal_processing import SignalProcessor, wavelet_denoising


class AudioProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.signal_processor = SignalProcessor(fs)

    def create_audio_dataframe(self, audio_file, respiration_file=None, file_path=None, target_sampling_rate=16000):
        audio = AudioSegment.from_file(audio_file)
        raw_data = audio.raw_data
        channels = audio.channels
        original_sampling_rate = audio.frame_rate
        audio_array = np.frombuffer(raw_data, dtype=np.int16)

        print(f'Original Sapling Rate = {original_sampling_rate}')
        print(f'Target Sapling Rate = {target_sampling_rate}')

        # Down-sample the audio array to the target sampling rate using librosa
        if original_sampling_rate != target_sampling_rate:
            # librosa expects mono audio, so reshape if necessary
            if channels > 1:
                audio_array = audio_array.reshape((-1, channels))
                audio_array = audio_array.mean(axis=1)  # Convert to mono by averaging channels

            # Resample the audio
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=original_sampling_rate,
                                           target_sr=target_sampling_rate)
            sampling_rate = target_sampling_rate
        else:
            sampling_rate = original_sampling_rate

        # Recalculate time vector based on the new (or unchanged) sampling rate
        time_v = np.arange(len(audio_array)) / sampling_rate

        audio_data = pd.DataFrame({'time': time_v, 'raw_data': audio_array})

        # --------------- Audio processing steps ----------------

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

        # --------------- end of audio processing steps ----------------
        if respiration_file:
            respiration_data = pd.read_csv(respiration_file)
            audio_data['Respiration'] = np.interp(audio_data.time, respiration_data['Data Set 1:Time(s)'],
                                                  respiration_data['Data Set 1:Force(N)'])

        if file_path:

            filename = os.path.splitext(os.path.basename(audio_file))[0]
            results_dir = os.path.join(file_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, filename + '_df.csv')
            audio_data.to_csv(save_path)
            print(f'File saved to: {save_path}')
        return audio_data, sampling_rate

    def respiration_shift(self, audio_df, distance=0, width=0):
        a = audio_df['wavelet_smoothed'].values
        b = audio_df['Respiration'].values
        y = signal.correlate(b, a)
        t = np.arange(-a.shape[0], a.shape[0] - 1) / self.fs
        p = signal.find_peaks(y, distance=distance, width=width)[0]
        shift = t[p[np.argmin(np.abs(t[p]))]]
        return audio_df['time'].values - shift
