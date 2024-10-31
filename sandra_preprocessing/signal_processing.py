import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, sosfiltfilt, tf2sos, iirnotch, sosfreqz
import noisereduce as nr


def wavelet_denoising(signal, threshold, wavelet='db4', level=None):
    def soft_thresholding(coef, thresh):
        return np.sign(coef) * np.maximum(np.abs(coef) - thresh, 0)

    coefficients = pywt.wavedec(signal, wavelet, level=level)
    threshold_value = np.percentile(np.abs(np.concatenate(coefficients)), threshold)
    soft_thresholded_coefficients = [soft_thresholding(c, threshold_value) for c in coefficients]
    denoised_signal = pywt.waverec(soft_thresholded_coefficients, wavelet)
    return denoised_signal[:signal.shape[0]]


class SignalProcessor:
    def __init__(self, fs):
        self.fs = fs

    def bandpass_filter(self, df, lowcut, highcut, order, notch=False, notch_freq=50, q=120, plot=False, xmax=None):
        nyquist_freq = 0.5 * self.fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # Validate frequencies
        if lowcut >= highcut:
            raise ValueError("Lowcut frequency must be lower than highcut frequency.")
        if lowcut <= 0 or highcut >= nyquist_freq:
            raise ValueError("Lowcut and highcut frequencies must be between 0 and Nyquist frequency.")

        # Design bandpass filter using second-order sections
        sos = butter(order, [low, high], btype='band', output='sos')

        # Apply bandpass filter using sosfiltfilt for better numerical stability
        df_filt = pd.DataFrame()
        for column in df:
            df_filt[column] = sosfiltfilt(sos, df[column])

        # Bandpass filter frequency response using sosfreqz
        w_bp, h_bp = sosfreqz(sos, worN=self.fs)
        frequency_bp = (nyquist_freq / np.pi) * w_bp

        freq_response_fig = None
        if notch:
            # Adjust notch frequency calculation for stability
            w0 = notch_freq / (self.fs / 2)  # Normalized frequency

            # Design notch filter
            b, a = iirnotch(w0, q)
            sos_notch = tf2sos(b, a)  # Convert to second-order sections

            # Apply notch filter using sosfiltfilt for better numerical stability
            for column in df_filt:
                df_filt[column] = sosfiltfilt(sos_notch, df_filt[column])

            # Combined frequency response
            w_notch, h_notch = sosfreqz(sos_notch, worN=self.fs)
            h_comb = h_bp * h_notch

            if plot:
                # Plot combined frequency response
                freq_response_fig = plt.figure(figsize=(10, 3))
                plt.plot(frequency_bp, 20 * np.log10(abs(h_comb + 1e-10)))
                plt.title("Combined Bandpass and Notch Filter Frequency Response")
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Gain (dB)')
                plt.xticks(np.arange(0, max(frequency_bp), 25))
                plt.grid()
                plt.axhline(y=-3, color='r', linestyle='--')  # -3 dB line

        else:
            if plot:
                # Plot bandpass frequency response
                freq_response_fig = plt.figure(figsize=(10, 3))
                plt.plot(frequency_bp, 20 * np.log10(abs(h_bp)), 'b')
                plt.title("Butterworth Bandpass Filter Frequency Response")
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Gain (dB)')
                plt.grid()
                plt.axhline(y=-3, color='r', linestyle='--')  # -3 dB line

        if plot:
            plt.ylim(-30, 1)
            if xmax is None:
                plt.xlim(0, highcut + highcut * 0.1)
            else:
                plt.xlim(0, xmax)
            plt.yticks(np.arange(-30, 3, step=3))

        if plot:
            return df_filt, freq_response_fig
        return df_filt

    def smoothed_signal(self, signal, window_size, iterations):
        window = int(self.fs * window_size / 1000)  # window size in ms
        demeaned = signal - signal.mean()
        rectified = demeaned.abs()
        smoothed = rectified
        for _ in range(iterations):
            smoothed = smoothed.rolling(window=window).mean()
        return smoothed.fillna(0)
