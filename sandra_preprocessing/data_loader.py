import os
from glob import glob
from .audio_processing import AudioProcessor


class DataLoader:
    def __init__(self, fs, folder_path):
        self.fs = fs
        self.folder_path = folder_path
        self.audio_processor = AudioProcessor(fs)

    def create_audio_dict(self, subjects, distances, resps, sampling_f, file_path):
        audio_dict = {}
        sampling_rate = self.fs
        for subject in subjects:
            folders = os.listdir(os.path.join(self.folder_path, subject))
            audio_dict[subject] = {}

            for distance in distances:
                audio_dict[subject][distance] = {}
                folder = folders[[i for i in range(len(folders)) if '(' + str(distance) + 'cm)' in folders[i]][0]]
                audio_files = glob(os.path.join(self.folder_path, subject, folder, '*cm.m*'))
                respiration_files = glob(os.path.join(self.folder_path, subject, folder, '*.csv'))

                for resp in resps:
                    audio_dict[subject][distance][resp] = {}

                    try:
                        # Find audio file
                        try:
                            af = [f for f in audio_files if resp in f.split('\\')[-1].lower()][0]
                        except IndexError:
                            print(f"Audio file for {resp} not found for {subject} at {distance} cm.")
                            continue

                        # Find respiration file
                        try:
                            rf = [f for f in respiration_files if resp in f.split('\\')[-1].lower()][0]
                        except IndexError:
                            print(f"Respiration file for {resp} not found for {subject} at {distance} cm.")
                            rf = None  # Set to None if respiration file is not found
                        af_name = af.split('\\')[-1]
                        rf_name = rf.split('\\')[-1]
                        # print(af)
                        print(f'processing: {subject} | {distance} | {resp}: {af_name}, {rf_name}')
                        # Create audio DataFrame using AudioProcessor class
                        audio_data, sampling_rate = self.audio_processor.create_audio_dataframe(af, rf,
                                                                                                file_path=file_path,
                                                                                                target_sampling_rate=sampling_f)

                        # Shift respiration using AudioProcessor's respiration_shift method
                        if rf:
                            audio_data['Respiration_time'] = self.audio_processor.respiration_shift(
                                audio_data, distance=sampling_rate, width=sampling_rate - 8
                            )

                        audio_dict[subject][distance][resp] = audio_data

                    except Exception as e:
                        print(f"Unexpected error occurred for {subject} at {distance} cm during {resp} processing: {e}")

        return audio_dict, sampling_rate
