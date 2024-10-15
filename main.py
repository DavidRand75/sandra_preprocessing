from sandra_preprocessing.data_loader import DataLoader
from sandra_preprocessing.plots import Plotter
from itertools import product

# Set up your parameters

data_path = 'G:\My Drive\Work\Sandra\Second Pilot - acoustic room\Data'
fs = 44100

# Initialize:
data_loader = DataLoader(fs, data_path)
plotter = Plotter(fs)

# Define subjects, distances, and respiration types

#subjects = ['Itai', 'Maor', 'David']
#distances = [2.5, 5, 10, 20]
#resps = ['shallow', 'deep']

subjects = ['Itai']
distances = [2.5]
resps = ['deep']

# Create the audio dictionary
audio_dict, _ = data_loader.create_audio_dict(subjects, distances, resps, fs, file_path=None)

for distance, resp, subject in product(distances, resps, subjects):
    audio_df = audio_dict[subject][distance][resp][1000:]
    Title = f'{subject} {str(distance)} cm {resp}'

    x = plotter.audio_fig(audio_df,
                          process_stage='raw_data',
                          title=Title,
                          save_path=data_path,
                          nfft=1024,
                          vmin=None,
                          vmax=None,
                          show_fig=True)


