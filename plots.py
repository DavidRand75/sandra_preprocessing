import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy.signal import spectrogram


class Plotter:
    def __init__(self, fs):
        self.fs = fs

    def plot_signal(self, signal, title, save_path, xmin='min', xmax='max', ymin='min', ymax='max', additional_df=None,
                    figsize=(10, 10), save=False):
        time_v = np.arange(len(signal)) / self.fs

        if xmax == 'max':
            xmax = max(time_v)
        if xmin == 'min':
            xmin = min(time_v)

        # Determine the number of subplots
        num_subplots = len(signal.columns)

        if additional_df is not None:
            num_subplots += 1

        fig, axs = plt.subplots(nrows=num_subplots, ncols=1, figsize=figsize)
        axs = np.atleast_1d(axs)

        for i, (column_label, column_data) in enumerate(signal.items()):
            axs[i].plot(time_v, column_data, linewidth=0.5)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Amplitude')
            axs[i].set_title(column_label)
            axs[i].set_xlim(xmin=xmin, xmax=xmax)
            if ymin != 'min' and ymax != 'max':
                axs[i].set_ylim(ymin=ymin, ymax=ymax)

        # Plot additional_df if provided
        if additional_df is not None:
            time_column = additional_df.columns[0]
            data_column = additional_df.columns[1]
            axs[-1].plot(additional_df[time_column], additional_df[data_column], linewidth=0.5, color='r')
            axs[-1].set_xlabel(time_column)
            axs[-1].set_ylabel(data_column)
            axs[-1].set_title(f'{data_column} vs {time_column}')
            if xmax == 'max':
                xmax = max(time_column)
            if xmin == 'min':
                xmin = min(time_column)
            axs[-1].set_xlim(xmin=xmin, xmax=xmax)

        plt.tight_layout()
        fig.suptitle(title)
        plt.show()

        if save:
            fig.savefig(os.path.join(save_path, title + '_plot.jpg'))

    def spectrograma(self, signal, title, save_path, nfft=1024, vmin=None, vmax=None, save=False):
        f = plt.figure(figsize=(10, 6))

        # Compute the spectrogram using scipy
        freqs, bins, sxx = spectrogram(signal, self.fs, nperseg=nfft, noverlap=int(nfft / 4))

        # Convert to dB scale, apply a threshold to avoid log(0)
        sxx = 10 * np.log10(np.maximum(sxx, 1e-10))  # Replace zero or negative values with 1e-10

        # Plot the spectrogram
        im = plt.pcolormesh(bins, freqs, sxx, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='dB/Hz')

        # Set plot limits, labels, and title
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(title)
        plt.xlim(0, max(bins))

        # Show the plot
        plt.show()

        # Save the figure if requested
        if save:
            f.savefig(os.path.join(save_path, title + '_spectogram.jpg'))

        return sxx, freqs, bins, im

    def audio_fig(self, audio_df, process_stage, title, nfft=1024, vmin=None, vmax=None, save_path=None, show_fig=False):
        # Create a figure with two horizontal subplots (1 row, 2 columns)
        audio_sig = audio_df[process_stage]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot the line graphs in the second subplot (ax2)
        ax1_1 = ax1.twinx()  # Second y-axis
        sns.lineplot(ax=ax1, data=audio_df, x='time', y='wavelet')
        sns.lineplot(ax=ax1_1, data=audio_df, x='time', y='Respiration', color='black', label='Respiration')
        sns.lineplot(ax=ax1_1, data=audio_df, x='Respiration_time', y='Respiration', color='red',
                     label='Respiration (shifted)').set(title=title)

        # Set labels for the second plot

        ax1.set_ylabel('Amplitude')
        ax1_1.set_ylabel('Respiration')

        # Compute the spectrogram using scipy
        freqs, bins, sxx = spectrogram(audio_sig, self.fs, nperseg=nfft, noverlap=int(nfft / 4))

        # Convert to dB scale, apply a threshold to avoid log(0)
        sxx = 10 * np.log10(np.maximum(sxx, 1e-10))  # Replace zero or negative values with 1e-10

        if process_stage == 'raw_data':
            freq_mask = freqs <= self.fs / 2 * 0.9
            filtered_sxx = sxx[freq_mask, :]
        else:
            freq_mask = (freqs <= 5000) & (freqs >= 70)
            filtered_sxx = sxx[freq_mask, :]

        if vmin is None:
            vmin = np.min(filtered_sxx)
        if vmax is None:
            vmax = np.max(filtered_sxx)

        # Plot the spectrogram in the first subplot (ax1)
        im = ax2.pcolormesh(bins, freqs, sxx, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        if process_stage == 'raw_data':
            ax2.set_ylim(0, self.fs / 2 * 0.9)
        else:
            ax2.set_ylim(70, 5000)

        plt.colorbar(im, ax=ax2, label='dB/Hz', orientation='horizontal', fraction=.06)

        # Set plot limits, labels, and title for the spectrogram
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Spectrogram')

        min_time = min(audio_df['time'].min(), min(bins))
        max_time = max(audio_df['time'].max(), max(bins))
        ax1.set_xlim(min_time, max_time)
        ax2.set_xlim(min_time, max_time)

        # Adjust layout
        plt.tight_layout()
        if save_path:
            figures_dir = os.path.join(save_path, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            save_dir = os.path.join(figures_dir, title + '_lineplot_spectrogram.jpg')
            fig.savefig(save_dir)
            print(f'Figure saved to: {save_dir}')
        # Show the combined plot
        if show_fig:
            plt.show()

        # Save the figure if requested

        return sxx, freqs, bins, im
