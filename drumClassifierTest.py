import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import librosa.display
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import glob
import os
import joblib
import soundfile as sf
import random
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import simpleaudio as sa

model_path = 'model/model.joblib'
scaler_path = 'model/scaler.joblib'

categories = ['kick', 'snare', 'open_hat', 'closed_hat', 'other']

# DrumClassifier (Working Title)
# by Prxybb
# apr 2023
# syntax & debug assistance by GPT-4

# Function to extract features from audio data
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    return np.mean(mfccs.T, axis=0)

def train_model():
    # Load and preprocess the dataset
    data = []
    labels = []
    
    folders = ['drumSortedData/loops1-10_sorted/kick', 'drumSortedData/loops1-10_sorted/snare', 'drumSortedData/loops1-10_sorted/open_hat', 'drumSortedData/loops1-10_sorted/closed_hat', 'drumSortedData/loops1-10_sorted/other']

    for label, folder in enumerate(folders):
        files = glob.glob(os.path.join(folder, "*.wav"))
        print(f"Loaded {len(files)} samples for class '{folder}'")
        for file in files:
            try:
                audio, sample_rate = librosa.load(file, sr=None)
                if len(audio) > sample_rate:
                    padded_audio = audio[:sample_rate]
                else:
                    padded_audio = np.zeros(sample_rate)
                    padded_audio[:len(audio)] = audio

                features = extract_features(padded_audio, sample_rate)
                data.append(features)
                labels.append(label)
            except (audioread.exceptions.NoBackendError, soundfile.LibsndfileError) as e:
                print(f"Skipping file '{file}' due to an error: {e}")

    if len(data) > 0:
        # Create a DataFrame
        df = pd.DataFrame(data)
        df['label'] = labels

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the SVM model
        model = SVC()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        return model, scaler
    else:
        print("No samples were loaded. Please check your folder names and file extensions.")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    user_input = input("Do you want to re-train the model? (y/N): ")
    if user_input.lower() == 'y':
        # Train your model here and save it again
        model, scaler = train_model()
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    else:
        # Load the existing model
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
else:
    # Train your model here if the files don't exist
    model, scaler = train_model()
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def classify_audio(file_path, model, scaler):
    # Load and pad the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    padded_audio = np.zeros(sample_rate)
    if len(audio) > sample_rate:
        padded_audio = audio[:sample_rate]
    else:
        padded_audio[:len(audio)] = audio

    # Extract features and normalize
    features = extract_features(padded_audio, sample_rate)
    normalized_features = scaler.transform([features])

    # Predict the class using the trained model
    predicted_class = model.predict(normalized_features)
    
    return predicted_class[0]

def classify_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Assuming audio files are in .wav format
            file_path = os.path.join(folder_path, filename)
            classification = classify_audio(file_path, model, scaler)
            print(f"{filename:<40} {categories[classification]}")

def sort_samples(input_folder, output_folder, batch_name="", sort_loudness=False):
    num_mfcc = 20

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if batch_name:
        parent_folder = os.path.join(output_folder, batch_name)
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
    else:
        parent_folder = find_next_folder_number(output_folder)

    for category in categories:
        category_folder = os.path.join(parent_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

    file_data = []

    for file in glob.glob(input_folder + '/*.wav'):
        file_path = file
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        mfccs_scaled = scaler.transform([mfccs_processed])
        category_id = model.predict(mfccs_scaled)[0]
        rms = librosa.feature.rms(y=audio)[0].mean()

        file_counter = 1
        while True:
            unsorted = ""
            if sort_loudness:
                unsorted = "_unsorted"
            new_file_name = f"{categories[category_id]}{str(file_counter).zfill(2)}-{batch_name}{unsorted}.wav"
            dest_file_path = os.path.join(parent_folder, categories[category_id], new_file_name)
            if not os.path.exists(dest_file_path):
                shutil.copy(file_path, dest_file_path)
                break
            file_counter += 1

        file_data.append((rms, category_id, file_path, new_file_name))

    if sort_loudness:
        file_data.sort(key=lambda x: x[0], reverse=True)
        print(file_data)

        category_ids = [0, 0, 0, 0, 0]

        for idx, (rms, category_id, file_path, new_file_name) in enumerate(file_data):
            category_ids[category_id] += 1
            os.remove(os.path.join(parent_folder, categories[category_id], new_file_name))
            new_file_name = f"{categories[category_id]}{str(category_ids[category_id]).zfill(2)}-{batch_name}.wav"
            shutil.copy(file_path, os.path.join(parent_folder, categories[category_id], new_file_name))
            print(f"{new_file_name}: {rms}")

def split_drum_loop(drum_loop_path, output_folder, prefix='drum_hit'):
    # Load the drum loop
    audio, sample_rate = librosa.load(drum_loop_path, sr=None)

    # Detect drum hits (onsets)
    onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sample_rate)

    # Pad the onset times with the end time of the audio to include the last drum hit
    onset_times = np.concatenate((onset_times, [len(audio) / sample_rate]))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    parent_folder = os.path.join(output_folder, prefix)

    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Split the drum loop into one-shots
    for i, start_time in enumerate(onset_times[:-1]):
        end_time = onset_times[i + 1]

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        drum_hit = audio[start_sample:end_sample]

        # Save the drum hit to a separate file
        output_file = os.path.join(parent_folder, f"{prefix}_{str(i + 1).zfill(3)}.wav")
        sf.write(output_file, drum_hit, sample_rate)

    print(f"Split drum loop into {len(onset_times) - 1} one-shot files in {output_folder}/")

def split_all_drum_loops(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            drum_loop_path = os.path.join(input_folder, file)
            split_drum_loop(drum_loop_path, output_folder, prefix=file[:-4])

def play_sample(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    audio = np.int16(audio * 32767)
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()

def manual_sorting(unsorted_folder, categories_folders):
    # Create the main window
    root = tk.Tk()
    root.title("Manual Drum Sample Sorting")

    for category_folder in categories_folders:
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

    def get_next_sample(unsorted_folder):
        unsorted_samples = [file for file in os.listdir(unsorted_folder) if os.path.isfile(os.path.join(unsorted_folder, file))]
        if unsorted_samples:
            return os.path.join(unsorted_folder, random.choice(unsorted_samples))
        else:
            return None

    def sort_sample(category):
        nonlocal current_sample_path
        destination_folder = categories_folders[category - 1]
        destination_file = os.path.join(destination_folder, os.path.basename(current_sample_path))
        os.rename(current_sample_path, destination_file)
        current_sample_path = get_next_sample(unsorted_folder)
        if current_sample_path is not None:
            play_sample(current_sample_path)
        else:
            print("No more samples to sort.")
            root.destroy()

    def on_key_press(event):
        key = event.char
        if key in ['1', '2', '3', '4', '5']:
            sort_sample(int(key))
        elif key == ' ':
            if current_sample_path is not None:
                play_sample(current_sample_path)

    root.bind('<KeyPress>', on_key_press)

    # Initial sample playback
    current_sample_path = get_next_sample(unsorted_folder)
    if current_sample_path is not None:
        play_sample(current_sample_path)

    root.mainloop()

# Basic Classification Usage 1:
# folder_path contains unsorted, digital drum one-shot samples
# classify_files_in_folder() classifies the samples into 5 drum group,
# and prints them out as a list

folder_path = "drumTestingData"
classify_files_in_folder(folder_path)

# Basic Classification Usage 1:
# folder_path contains unsorted, digital drum one-shot samples
# output_folder_path is an available empty directory
# batch_name sets a batch name subfolder/prefix to identify the samples
# sort_loudness sorts the samples from loudest to quietest within their classified folder
# sort_samples() classifies the samples into 5 drum group subfolders

folder_path = "drumTestingData"
output_folder_path = "drumSortedData/"
batch_name = "sortingExample"
sort_loudness = True
sort_samples(folder_path, output_folder_path, batch_name=batch_name, sort_loudness=sort_loudness)
