import numpy as np
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt, resample
import matplotlib.pyplot as plt
import librosa
import librosa.display
import crepe
import os
from IPython.display import Audio
import csv
import os.path
import plotly.graph_objects as go
import sounddevice as sd

log_frequencies = [4.0313, 4.1146, 4.1979, 4.2812, 4.3645, 4.4478, 4.5311, 4.6144, 4.6977, 4.781, 4.8643, 4.9476, 5.0309, 
                   5.1142, 5.1975, 5.2808, 5.3641, 5.4474, 5.5307, 5.614, 5.6973, 5.7806, 5.8639, 5.9472, 6.0305, 6.1138, 
                   6.1971, 6.2804, 6.3637, 6.447, 6.5303, 6.6136, 6.6969, 6.7802, 6.8635, 6.9468, 7.0301, 7.1134, 7.1967, 
                   7.28, 7.3633, 7.4466, 7.5299, 7.6132, 7.6965, 7.7798, 7.8631, 7.9464, 8.0297, 8.113, 8.1963, 8.2796, 
                   8.3629, 8.4462, 8.5295, 8.6128, 8.6961, 8.7794, 8.8627, 8.946, 9.0293, 9.1126, 9.1959, 9.2792, 9.3625, 
                   9.4458 , 9.5291, 9.6124, 9.6957, 9.779, 9.8623, 9.9456, 10.0289, 10.1122, 10.1955, 10.2788, 10.3621, 
                   10.4454, 10.5287, 10.612, 10.6953, 10.7786, 10.8619, 10.9452, 11.0285, 11.1118, 11.1951, 11.2784, 
                   11.3617, 11.445, 11.5283, 11.6116, 11.6949, 11.7782, 11.8615, 11.9448, 12.0281]
def mute_crepe_frequencies(audio_path, output_file="crepe_output.txt", offset=0.0, duration=None):
    # Load the audio
    y, sr = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

    # Check if CREPE output exists
    if os.path.exists(output_file):
        # Load saved frequency data
        frequency = []
        with open(output_file, "r") as f:
            for line in f:
                if not line.strip():  # Ignore empty lines
                    continue
                frequency.append(float(line.strip()))
        frequency = np.array(frequency)

        # Estimate corresponding time array based on signal length and CREPE default hop size
        time = np.linspace(0, len(y) / sr, len(frequency))
        print("Loaded CREPE frequency output from file.")
    else:
        # Run CREPE to get pitch
        time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)

        # Save the frequency data to a file
        with open(output_file, "w") as f:
            for freq in frequency:
                f.write(f"{freq}\n")
        print("CREPE output calculated and saved to file.")

    # Compute the spectrogram
    D = librosa.stft(y)
    S = np.abs(D)  # Magnitude spectrogram
    S_min = np.min(S)  # Minimum value in the spectrogram for muting

    # Map time to spectrogram frames
    stft_time = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    time_frames = np.searchsorted(stft_time, time)

    # Clip time indices to avoid out-of-bounds errors
    time_frames = np.clip(time_frames, 0, S.shape[1] - 1)

    # Get frequency bins
    freq_bins = librosa.fft_frequencies(sr=sr)

    # Initialize a list to track the muted harmonics
    muted_harmonics = []

    # Mute the frequencies and their harmonics
    for t_idx, f in zip(time_frames, frequency):
        if f > 0:  # Ignore zero frequencies
            # Iterate through multiple harmonics
            h = 1
            while f * h < np.max(freq_bins):  # Stop when harmonics exceed max spectrogram frequency
                harmonic_freq = f * h
                bin_idx = np.argmin(np.abs(freq_bins - harmonic_freq))
                if bin_idx < S.shape[0]:  # Ensure index is within bounds
                    # Mute the harmonic frequency in the spectrogram for all time frames
                    S[bin_idx, t_idx] = S_min
                    # Record the muted harmonic
                    muted_harmonics.append(harmonic_freq)
                h += 1

    # Convert to dB for display
    S_db_original = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S_db_muted = librosa.amplitude_to_db(S, ref=np.max)

    # Plot the original and muted spectrograms
    plt.figure(figsize=(16, 10))

    # Original spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_db_original, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Muted spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_db_muted, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Muted Spectrogram (Fundamentals and All Harmonics Removed)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

    # Return the muted spectrogram and the harmonics that were muted
    return S_db_muted # np.array(muted_harmonics)


def Play_spectrogram(spectrogram_array, sr=16000):
    """
    Plays the audio reconstructed from a spectrogram array using librosa and IPython.
    
    Parameters:
    - spectrogram_array: Spectrogram to be played (in dB or magnitude form)
    - sr: Sampling rate (default 16000)
    """
    # Convert the spectrogram back to amplitude
    S_amp = librosa.db_to_amplitude(spectrogram_array)  # Convert from dB to amplitude
    
    # If you only have magnitude, we need phase information to reconstruct the audio.
    # Let's create a random phase (phase information was lost in muting harmonics)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(*S_amp.shape))  # Random phase
    
    # Reconstruct the complex spectrogram
    S_complex = S_amp * phase  # Combine magnitude with random phase
    
    # Perform ISTFT (Inverse Short-Time Fourier Transform)
    audio_signal = librosa.istft(S_complex)

    # Play the reconstructed audio signal using IPython display
    return Audio(audio_signal, rate=sr)
def sum_harmonics_amplitude(audio_path, sr=16000, offset=0.0, duration=None):
    # Load the audio
    y, sr = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)

    # Extract the harmonic component of the audio signal
    y_harmonic, _ = librosa.effects.hpss(y)

    # Perform STFT to get the frequency spectrum
    D = librosa.stft(y_harmonic)
    S = np.abs(D)  # Magnitude spectrogram
    freq_bins = librosa.fft_frequencies(sr=sr)

    # Define a dictionary to store the sum of amplitudes for each note
    note_amplitudes = {}

    # Define the frequency ranges for each note and its harmonics (up to the 6th harmonic)
    note_frequencies = {}
    for note in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
        note_frequencies[note] = []
        for i in range(1, 7):  # Harmonics from 1st to 6th
            note_frequencies[note].append(librosa.note_to_hz(f'{note}{i}'))

    # Iterate through each note (A, B, C, etc.)
    for note, harmonics in note_frequencies.items():
        total_amplitude = 0
        # For each harmonic of the note, sum the amplitudes of its corresponding frequency bins
        for harmonic in harmonics:
            # Find indices of frequency bins corresponding to the harmonic
            harmonic_indices = np.where((freq_bins >= harmonic - 5) & (freq_bins <= harmonic + 5))[0]  # ±5 Hz tolerance

            # Sum the amplitudes of the bins around this harmonic
            total_amplitude += np.sum(S[harmonic_indices, :])

        # Store the summed amplitude in the dictionary
        note_amplitudes[note] = total_amplitude

    # Plotting the spectrogram for visual reference
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    # Return the dictionary of summed amplitudes for each note
    return note_amplitudes
def plot_crepe_pitch(audio_path, output_file="crepe_output.csv", offset=0.0, duration=None):
    """
    Plots the pitch contour extracted using CREPE, along with its first and second derivatives.

    Parameters:
    - audio_path: Path to the audio file.
    - output_file: File to save/load CREPE output.
    - offset: Starting point of the audio (in seconds).
    - duration: Duration of the audio to analyze (in seconds).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import os
    import csv
    import crepe

    # Load the audio
    y, sr = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

    # Generate time array for the audio
    audio_duration = len(y) / sr
    time = np.linspace(0, audio_duration, num=len(y))

    # Check if CREPE output exists
    if os.path.exists(output_file):
        # Load saved frequency data from CSV
        with open(output_file, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            data = np.array([[float(value) for value in row] for row in csvreader])

        time = data[:, 0]
        frequency = data[:, 1]
        confidence = data[:, 2]
        activation = data[:, 3]
        print("Loaded CREPE frequency output from CSV file.")
    else:
        # Run CREPE to get pitch
        time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)

        # Save the frequency data to a CSV file
        with open(output_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the header
            csvwriter.writerow(["Time", "Frequency", "Confidence", "Activation"])

            # Write data rows
            for t, freq, conf, act in zip(time.flatten(), frequency.flatten(), confidence.flatten(), activation.flatten()):
                csvwriter.writerow([t, freq, conf, act])

        print("CREPE output calculated and saved to CSV file.")

    # Compute first and second derivatives
    first_derivative = np.gradient(frequency)
    second_derivative = np.gradient(first_derivative)

    # Normalize the derivatives for better visualization
    first_derivative_norm = first_derivative / np.max(np.abs(first_derivative))
    second_derivative_norm = second_derivative / np.max(np.abs(second_derivative))

    # Plot the pitch contour and its derivatives
    plt.figure(figsize=(14, 8))

    # Plot original frequency contour
    plt.plot(time, frequency, color='b', linewidth=1.5, label='Frequency (Hz)')

    # Plot first derivative
    plt.plot(time, first_derivative_norm * 500, color='g', linestyle='--', linewidth=1.5, label='1st Derivative (Normalized)')

    # Plot second derivative
    plt.plot(time, second_derivative_norm * 500, color='r', linestyle=':', linewidth=1.5, label='2nd Derivative (Normalized)')

    # Add title, labels, and legend
    plt.title('CREPE Pitch Contour and Derivatives')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz) and Derivatives (Scaled)')
    plt.ylim(0, 2000)  # Limit y-axis for better visibility
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return frequency, confidence
def evaluate_segments(segments):
    # Flatten the segments list to get a list of tuples
    flattened_segments = [point for segment in segments for point in segment]
    
    # Calculate the number of points
    num_points = len(flattened_segments)
    
    # Calculate the number of notes
    num_notes = len(segments)
    
    # Calculate the average standard deviation of each note
    avg_std = np.mean([np.std([point[1] for point in segment]) for segment in segments])
    
    return num_points, num_notes, avg_std
def find_best_parameters(log_f0, log_frequencies):
    best_params = None
    best_score = float('-inf')
    
    for threshold in np.arange(0, 1, 0.1):
        for ratio in np.arange(0.5, 0.9, 0.1):
            for tangent_threshold in np.arange(0, 1, 0.1):
                segments = get_notes(log_f0, threshold, ratio, tangent_threshold)
                num_points, num_notes, avg_std = evaluate_segments(segments)
                
                # Calculate a score based on the three criteria
                # We want to minimize num_notes, maximize average segment length, and minimize avg_std
                # Score = -num_notes + (average_length * weight) - (avg_std * penalty)
                
                average_length = np.mean([len(segment) for segment in segments]) if segments else 0
                score = -num_notes + (average_length * 0.1) - (avg_std * 10)  # Adjust weights as needed
                
                if score > best_score:
                    best_score = score
                    best_params = (threshold, ratio, tangent_threshold)
    print(best_params)
    return best_params
def get_notes(aligned_log_f0, threshold=1.0, ratio=0.7666, tangent_threshold=0.2):
    time_values = np.arange(len(aligned_log_f0))
    segments = []
    current_segment = []

    # Segment the frequency data
    for i in range(len(aligned_log_f0)):
        if i == 0:
            current_segment.append((time_values[i], aligned_log_f0[i]))
        else:
            mean_value = np.mean([point[1] for point in current_segment])
            if abs(aligned_log_f0[i] - mean_value) <= threshold:
                current_segment.append((time_values[i], aligned_log_f0[i]))
            else:
                segments.append(current_segment)
                current_segment = [(time_values[i], aligned_log_f0[i])]

    # Append the last segment if it exists
    if current_segment:
        segments.append(current_segment)

    segment_lengths = [len(segment) for segment in segments]
    average_length = sum(segment_lengths) / len(segment_lengths)
    current_group = []

    # Merge short segments
    i = 0
    while i < len(segments):
        segment = segments[i]
        if len(segment) <= ratio * average_length:  # Short segment
            current_group.extend(segment)
            segments.pop(i)  # Remove the current segment
        else:
            if current_group:  # Merge the current group
                segments.insert(i, current_group)
                current_group = []
                i += 1  # Move to the next segment
            i += 1
    if current_group:
        segments.append(current_group)

    def calculate_tangent(segment):
        times, freqs = zip(*segment)
        coeffs = np.polyfit(times, freqs, 1)  # Fit a line (degree 1 polynomial)
        slope = coeffs[0]  # The slope is the tangent of the angle
        return abs(slope)

    # Remove very short segments or those with high tangent values
    segments = [segment for segment in segments if len(segment) > 2]
    segments = [segment for segment in segments if len(segment) >= 4 or calculate_tangent(segment) <= tangent_threshold]

    # Convert segments into time frames
    time_frames = [(segment[0][0], segment[-1][0]) for segment in segments]

    return time_frames











def crepe_initialize(y,sr, output_file="crepe_output.csv"):

    if os.path.exists(output_file):
        # Load saved frequency data
        data = np.genfromtxt(output_file, delimiter=',', skip_header=1)
        time_frames = data[:, 0]
        frequency = data[:, 1]
        confidence = data[:, 2]
        activation = data[:, 3]
        magnitude = data[:, 4]
        print("Loaded CREPE frequency output from file.")
    else:
        # Run CREPE to get pitch (frequencies, confidence, activation)
        time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
        
        # Prepare to calculate magnitudes
        magnitudes = []
        time_frames = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=512)

        # Calculate the magnitude for each frequency at the corresponding time
        for t, freq in zip(time, frequency):
            # Find the closest time frame
            frame_index = np.argmin(np.abs(time_frames - t))
            # Find the closest frequency bin
            freq_bin = np.argmin(np.abs(librosa.fft_frequencies(sr=sr) - freq))
            # Get the magnitude from the spectrogram
            mag = np.abs(S_db[freq_bin, frame_index])
            magnitudes.append(mag)

        # Save the time, frequencies, confidence, activation, and magnitude to the output file
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Frequency (Hz)", "Confidence", "Activation", "Magnitude"])  # Header
            for t, freq, conf, act, mag in zip(time, frequency, confidence, activation, magnitudes):
                writer.writerow([t, freq, conf, act, mag])  # Write time, frequency, confidence, activation, and magnitude
        print("CREPE output calculated and saved to file.")

    return output_file

def get_carnatic_frequencies(tonic):
    # Intonational ratios for the basic set of Carnatic notes
    carnatic_ratios = {
        'Sa': 1.0,    # Tonic (Sa)
        'Ri1': 16/5, # Ri1
        'Ri2': 9/8,  # Ri2
        'Ga1': 6/5,  # Ga1
        'Ga2': 5/4, # Ga2
        'Ma1': 4/3, # Ma1
        'Ma2': 45/32,   # Ma2
        'Pa': 3/2,    # Pa
        'Dha1': 8/5, # Dha1
        'Dha2': 5/3, # Dha2
        'Ni1': 16/9, # Ni1
        'Ni2': 15/8,   # Ni2
        'Sa2': 2.0,   # Octave higher (Sa)
    }

    tonic_freq = librosa.note_to_hz(tonic)  # Get the frequency of the tonic

    # Calculate the frequencies for each Carnatic note relative to the tonic
    carnatic_frequencies = {note: tonic_freq * ratio for note, ratio in carnatic_ratios.items()}
    return carnatic_frequencies

def plot_frequency_with_carnatic_notes(frequency_list, tonic='G#3', threshold=0.5):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.signal import find_peaks

    # Convert input to a NumPy array for easier processing
    frequency_array = np.array(frequency_list)

    # Identify valid (non-NaN) frames
    valid_indices = ~np.isnan(frequency_array)  
    valid_frequencies = frequency_array[valid_indices]
    if len(valid_frequencies) == 0:
        raise ValueError("No valid frequencies to process.")

    # Get Carnatic note frequencies based on the input tonic
    carnatic_frequencies = get_carnatic_frequencies(tonic)

    # Get indices of valid (non-NaN) frequencies
    time_indices = np.arange(len(frequency_array))[valid_indices]

    # Find peaks and valleys in the valid frequency data
    peaks, _ = find_peaks(valid_frequencies)
    valleys, _ = find_peaks(-valid_frequencies)

    # Map peak and valley indices back to original time indices
    peak_x = time_indices[peaks]
    valley_x = time_indices[valleys]

    # Plot the graph
    fig = go.Figure()

    # Plot the frequency graph with gaps for NaNs
    for start, end in zip(
        np.where(np.diff(np.concatenate(([0], valid_indices, [0]))) == 1)[0],
        np.where(np.diff(np.concatenate(([0], valid_indices, [0]))) == -1)[0]
    ):
        fig.add_trace(go.Scatter(
            x=np.arange(start, end),
            y=frequency_array[start:end],
            mode='lines',
            name='Frequency (Hz)',
            line=dict(color='blue')
        ))

    # Plot horizontal lines for Carnatic notes
    for note, freq in carnatic_frequencies.items():
        fig.add_trace(go.Scatter(
            x=[0, len(frequency_list) - 1],
            y=[freq, freq],
            mode='lines',
            line=dict(dash='dash', color='gray', width=2),
            name=note,
            hovertemplate=f"{note} ({freq:.2f} Hz)"
        ))

    # Plot peaks (highlighted yellow)
    fig.add_trace(go.Scatter(
        x=peak_x,
        y=frequency_array[peak_x],  # Corrected y-values using original indices
        mode='markers',
        name='Peaks',
        marker=dict(color='yellow', size=8),
        hovertemplate='Peak: %{y:.2f} Hz'
    ))

    # Plot valleys (highlighted red)
    fig.add_trace(go.Scatter(
        x=valley_x,
        y=frequency_array[valley_x],  # Corrected y-values using original indices
        mode='markers',
        name='Valleys',
        marker=dict(color='red', size=8),
        hovertemplate='Valley: %{y:.2f} Hz'
    ))

    fig.update_layout(
        title=f'Frequency with Carnatic Notes (Tonic: {tonic})',
        xaxis_title='Time',
        yaxis_title='Frequency (Hz)',
        showlegend=True
    )

    fig.show()








audio_path = r"C:\Desktop\Python\Audio Signal Processing\downloads\split sounds\Enna Thavam Seithanai I Sooryagayathri I Carnatic Krithi ｜ Papanasam Sivan\vocals.wav"
y, sr = librosa.load(audio_path, sr=16000, offset=20, duration=3)
y_slow = librosa.effects.time_stretch(y, rate=0.5)
audio_duration = len(y) / sr
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.min)

max_value = np.max(S_db)
threshold = (3/4) * max_value   
skeletal_S_db = np.where(S_db < threshold, 0, S_db)
    
print("playing")
# sd.play(y_slow, sr)
# sd.wait()
print("played")
output_file = "crepe_output.csv"

# crepe_initialize(y,sr, output_file)

with open(output_file, mode='r') as file:
    reader = list(csv.reader(file))
    time = [float(row[0]) for i, row in enumerate(reader) if i > 0]
    frequency = [float(row[1]) for i, row in enumerate(reader) if i > 0]
    confidence = [float(row[2]) for i, row in enumerate(reader) if i > 0]
    magnitude = [float(row[4]) for i, row in enumerate(reader) if i > 0]
if time[-1] > audio_duration:
    print("CREPE output time exceeds the segment duration. Please verify the CREPE processing.")


def plot_spectogram_with_crepe(time, frequency, S_db, sr):
    print("plotting")
    # Plot the spectrogram
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with CREPE Pitch Overlay')

    # Overlay the CREPE pitch line
    plt.plot(time, frequency, color='r', linewidth=1.5, label='CREPE Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 2000)  # Limit y-axis to 2000 Hz for better visibility

    # Use a fixed location for the legend
    plt.legend(loc='upper right')  # Place the legend explicitly in the upper-right corner

    plt.tight_layout()
    plt.show()
    print("plotted")

def check_confidence(frequency, confidence, threshold=0.8):
    """Return frequencies where confidence is above the specified threshold."""
    new_frequency = frequency.copy()

    # Filter based on confidence
    for i in range(len(confidence)):
        if confidence[i] < threshold: 
            new_frequency[i] = float('nan')

    return new_frequency


conf=check_confidence(frequency, confidence, threshold=0.6)
# plot_spectogram_with_crepe(time, conf, S_db, sr)



def get_harmonic_indices(fundamental_freq, sr, S_db):
    """Get indices of harmonics in the spectrogram."""
    harmonic_indices = []
    harmonic_number = 1
    while True:
        harmonic_freq = fundamental_freq * harmonic_number
        if harmonic_freq > sr / 2:  # Nyquist frequency
            break
        harmonic_bin = int(np.round(harmonic_freq * len(S_db) / sr))
        harmonic_indices.append(harmonic_bin)
        harmonic_number += 1
    return harmonic_indices


def calculate_hnr(S_db, frequency, sr):
    """Calculate the Harmonic-to-Noise Ratio (HNR) for each frame."""
    hnr_list = []
    
    for i, freq in enumerate(frequency):
        # Determine harmonic indices based on the fundamental frequency
        harmonic_indices = get_harmonic_indices(freq, sr, S_db[i])
        
        # Get the magnitudes of the harmonics for the specific frame
        harmonic_magnitudes = S_db[i][harmonic_indices]
        
        
        hnr_list.append(sum(harmonic_magnitudes[:8]))

    return hnr_list

def filter_frequencies_with_harmonics(S_db, frequency, confidence,hnr, sr, confidence_threshold=0.6,hnr_threshold=-3):
    filtered_frequencies = frequency.copy()
    for i, freq in enumerate(frequency):
        if confidence[i] < confidence_threshold:
            filtered_frequencies[i]=float('nan')
            continue
    
    # for i, freq in enumerate(frequency):
    #     print(hnr[i])
    #     if hnr[i] > hnr_threshold:
    #         print("yes")
    #         filtered_frequencies[i]=float('nan')
    #         continue
    
    return filtered_frequencies

def normalize_list(input_list):
    """Normalize a list so that each element lies between 0 and 1, ignoring NaN values."""
    # Create a filtered list that excludes NaN values
    filtered_list = [value for value in input_list if not np.isnan(value)]
    
    # If the filtered list is empty, return the original list
    if not filtered_list:
        return input_list

    min_value = min(filtered_list)
    max_value = max(filtered_list)
    
    # Avoid division by zero if all non-NaN values are the same
    if max_value == min_value:
        return [0.0 if np.isnan(value) else 1.0 for value in input_list]  # or return [1.0] * len(input_list) if you prefer

    # Normalize the original list, keeping NaN values unchanged
    normalized_list = [
        (value - min_value) / (max_value - min_value) if not np.isnan(value) else np.nan
        for value in input_list
    ]
    
    return normalized_list







import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import csv

# Load your audio file and process it
audio_path = r"C:\Desktop\Python\Audio Signal Processing\downloads\split sounds\Enna Thavam Seithanai I Sooryagayathri I Carnatic Krithi ｜ Papanasam Sivan\vocals.wav"
y, sr = librosa.load(audio_path, sr=16000, offset=20, duration=3)
audio_duration = len(y) / sr
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.min)

max_value = np.max(S_db)
threshold = (3/4) * max_value   
skeletal_S_db = np.where(S_db < threshold, 0, S_db)

output_file = "crepe_output.csv"

# Assuming you have already processed the audio with CREPE and saved the output
with open(output_file, mode='r') as file:
    reader = list(csv.reader(file))
    time = [float(row[0]) for i, row in enumerate(reader) if i > 0]
    frequency = [float(row[1]) for i, row in enumerate(reader) if i > 0]
    confidence = [float(row[2]) for i, row in enumerate(reader) if i > 0]



def frequencies_to_bins(frequencies, sr, num_bins):
    """Convert a list of frequencies to their corresponding frequency bin indices."""
    bin_indices = []
    for freq in frequencies:
        # Calculate the bin index
        bin_index = int(np.round(freq * num_bins / sr))
        bin_indices.append(bin_index)
    return bin_indices


def get_harmonic_indices(fundamental_freq, sr):
    overtones=[]
    for i in range (8):
        overtones.append(fundamental_freq*(i+1))
    harmonic_indices = frequencies_to_bins(overtones, sr, 1024)

    
    return harmonic_indices

def plot_spectrogram_with_harmonics(S_db, frequency, sr, frame_index):
    """Plot the spectrogram and mark harmonic bins for a specific frame."""
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with Harmonic Bins')

    # Extract the frequency for the specified frame
    fundamental_freq = frequency[frame_index]
    num_bins = S_db.shape[0]  # Get the number of frequency bins from S_db
    
    # Get harmonic indices
    harmonic_indices = get_harmonic_indices(fundamental_freq, sr, num_bins)
    print(harmonic_indices)
    # Plot the harmonic bins
    for harmonic_bin in harmonic_indices:
        if harmonic_bin < S_db.shape[0]:  # Ensure the bin is within bounds
            plt.scatter([time[frame_index]], [harmonic_bin], color='red', s=50, label='Harmonic Bin' if harmonic_bin == harmonic_indices[0] else "")

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 2000)  # Limit y-axis to 2000 Hz for better visibility
    plt.legend()
    plt.tight_layout()
    plt.show()



import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram_with_crepe_and_harmonics(time, frequency, S_db, sr, num_harmonics=8):
    print("Plotting...")
    
    # Plot the spectrogram
    plt.figure(figsize=(14, 8))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with CREPE Pitch and Harmonics Overlay')
    
    # Overlay the CREPE pitch line
    plt.plot(time, frequency, color='r', linewidth=1.5, label='CREPE Pitch')
    
    # Overlay the harmonics
    for h in range(2, num_harmonics + 2):  # First 8 harmonics
        harmonic_freq = frequency * h  # Calculate harmonic frequencies
        plt.plot(time, harmonic_freq, linestyle='dashed', linewidth=1, label=f'Harmonic {h}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, sr / 2)  # Set the limit to Nyquist frequency
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Plotted.")

plot_spectrogram_with_crepe_and_harmonics(time, frequency, S_db, sr, num_harmonics=8)

















