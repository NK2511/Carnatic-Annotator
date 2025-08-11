# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import random
import numpy as np
import pandas as pd
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

# Scientific & Audio Processing Libraries
import librosa
import crepe
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from fastdtw import fastdtw

# Machine Learning Libraries
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Plotting & Display
import matplotlib.pyplot as plt
import IPython.display as ipd

# ============================================================================
# CONFIGURATION
# ============================================================================

AUDIO_CONFIG = {
    "sample_rate": 44100,
    "confidence_threshold": 0.7
}

CREPE_CONFIG = {
    "viterbi": True,
    "step_size": 20,
    "model_capacity": "tiny",
    "verbose": False
}

CLUSTERING_CONFIG = {
    "initial_window_size": 60,
    "decay_size": 2,
    "min_window_size": 15,
    "hop_factor": 12,
    "outlier_threshold": 2,
    "similarity_threshold": 0.04,
    "pca_components": 15,
    "second_phase_window_size": 50,
    "similarity_threshold_secondary": 0.7,
    "pca_components_secondary": 10
}

CARNATIC_RATIOS = {
    # Lower Octave (Mandra Sthayi)
    'sa': 0.5, 'ri1': 0.5 * 16/15, 'ri2': 0.5 * 9/8, 'ga2': 0.5 * 6/5,
    'ga3': 0.5 * 5/4, 'ma1': 0.5 * 4/3, 'ma2': 0.5 * 45/32, 'pa': 0.5 * 3/2,
    'da1': 0.5 * 8/5, 'da2': 0.5 * 5/3, 'ni2': 0.5 * 16/9, 'ni3': 0.5 * 15/8,
    # Middle Octave (Madhya Sthayi)
    'Sa': 1.0, 'Ri1': 1.0 * 16/15, 'Ri2': 1.0 * 9/8, 'Ga2': 1.0 * 6/5,
    'Ga3': 1.0 * 5/4, 'Ma1': 1.0 * 4/3, 'Ma2': 1.0 * 45/32, 'Pa': 1.0 * 3/2,
    'Da1': 1.0 * 8/5, 'Da2': 1.0 * 5/3, 'Ni2': 1.0 * 16/9, 'Ni3': 1.0 * 15/8,
    # Upper Octave (Tara Sthayi)
    'SA': 2.0, 'RI1': 2.0 * 16/15, 'RI2': 2.0 * 9/8, 'GA2': 2.0 * 6/5,
    'GA3': 2.0 * 5/4, 'MA1': 2.0 * 4/3, 'MA2': 2.0 * 45/32, 'PA': 2.0 * 3/2,
    'DA1': 2.0 * 8/5, 'DA2': 2.0 * 5/3, 'NI2': 2.0 * 16/9, 'NI3': 2.0 * 15/8,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_paths(raaga_name: str) -> Dict[str, str]:
    """Get file paths for a raaga."""
    data_dir = Path("data") / raaga_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return {
        "master_csv": str(data_dir / f"master_{raaga_name}.csv"),
        "carva_csv": str(data_dir / f"carva_{raaga_name}.csv"),
        "log_csv": str(data_dir / f"log_{raaga_name}.csv")
    }

def get_closest_note(freq: float, carnatic_frequencies: Dict[str, float]) -> str:
    """Find the closest Carnatic note for a given frequency."""
    return min(carnatic_frequencies, key=lambda note: abs(carnatic_frequencies[note] - freq))

def get_closest_frequency(freq: float, carnatic_frequencies: Dict[str, float]) -> float:
    """Find the closest Carnatic note frequency for a given frequency."""
    return min(carnatic_frequencies.values(), key=lambda f: abs(f - freq))

def clean_np_float_list(seg_str: str) -> np.ndarray:
    """Convert a stringified list with np.float64 entries into a proper list of floats."""
    cleaned = re.sub(r'np\\.float64\\(([^)]+)\\)', r'\\1', seg_str)
    return np.array(ast.literal_eval(cleaned), dtype=float)

def interpolate_list(lst: list, target_len: int) -> list:
    """Interpolate a list of values to a target length, handling empty lists."""
    if lst is None or len(lst) == 0:
        return [0.0] * target_len
    original_len = len(lst)
    return list(np.interp(np.linspace(0, original_len - 1, target_len),
                          np.arange(original_len), lst))

# ============================================================================
# PHASE 1: DATA PREPARATION & INITIAL CLUSTERING
# ============================================================================

def process_audio_directory(audio_dir: str):
    """
    Processes all audio files in a directory to create a master CSV with pitch data.
    """
    print(f"Processing audio files in '{audio_dir}'...")
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    log_path, master_csv_path = paths["log_csv"], paths["master_csv"]

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    processed_files = set(pd.read_csv(log_path)["AudioPath"].values) if Path(log_path).exists() and Path(log_path).stat().st_size > 0 else set()
    if not processed_files:
        with open(log_path, "w") as f: f.write("AudioPath\n")

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    unprocessed_files = [f for f in audio_files if os.path.join(audio_dir, f) not in processed_files]

    if not unprocessed_files:
        print("✅ No new audio files to process.")
        return

    current_song_index = 0
    if Path(master_csv_path).exists() and Path(master_csv_path).stat().st_size > 0:
        existing_df = pd.read_csv(master_csv_path)
        if 'Index' in existing_df.columns and not existing_df.empty:
            current_song_index = existing_df["Index"].max() + 1
    else:
        with open(master_csv_path, "w") as f:
            f.write("Index,AudioPath,Raaga,SongName,Tonic,Time,Frequency,Confidence\n")

    print(f"🎵 Found {len(unprocessed_files)} new audio files to process, starting at index {current_song_index}")
    
    new_log_entries = []
    for filename in tqdm(unprocessed_files, desc="Processing audio"):
        audio_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(audio_path, sr=AUDIO_CONFIG["sample_rate"])
        
        name_without_ext = Path(filename).stem
        parts = name_without_ext.split("_")
        raaga, songname, tonic = (parts[0], "_".join(parts[1:-1]), parts[-1]) if len(parts) >= 3 else ("Unknown", "Unknown", "Unknown")
        
        time, frequency, confidence, _ = crepe.predict(y, sr, **CREPE_CONFIG)
        spec_time = librosa.times_like(librosa.stft(y), sr=sr)
        
        interp_freq = interp1d(time, frequency, kind='linear', fill_value='extrapolate')(spec_time)
        interp_conf = interp1d(time, confidence, kind='linear', fill_value='extrapolate')(spec_time)

        mask = interp_conf > AUDIO_CONFIG["confidence_threshold"]
        
        df_single = pd.DataFrame({
            "Index": current_song_index, "AudioPath": audio_path, "Raaga": raaga,
            "SongName": songname, "Tonic": tonic, "Time": spec_time[mask],
            "Frequency": interp_freq[mask], "Confidence": interp_conf[mask]
        })
        
        df_single.to_csv(master_csv_path, mode='a', index=False, header=False)
        new_log_entries.append(audio_path)
        current_song_index += 1
            
    if new_log_entries:
        pd.DataFrame({"AudioPath": new_log_entries}).to_csv(log_path, mode='a', index=False, header=False)
    
    print("\n🎉 Audio processing complete!")

def add_normalized_frequency_column(audio_dir: str):
    """
    Adds a 'Tonic_Normalized_Frequency' column to the master CSV file.
    """
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    df = pd.read_csv(paths["master_csv"])

    def get_tonic_freq(tonic_note):
        try: return librosa.note_to_hz(tonic_note)
        except (ValueError, TypeError): return np.nan

    tonic_map = {tonic: get_tonic_freq(tonic) for tonic in df["Tonic"].unique()}
    
    base_freqs = df["Tonic"].map(tonic_map)
    df["Tonic_Normalized_Frequency"] = df["Frequency"] / base_freqs
    
    df.to_csv(paths["master_csv"], index=False)
    print("✅ Tonic_Normalized_Frequency column added.")

def perform_multistage_clustering(audio_dir: str, config: Dict, song_index: Optional[int] = None):
    """
    Optimized clustering function using a distance matrix modification approach.
    """
    # This function is already well-placed and documented from the previous turn.
    # No changes to its internal logic are made, only reordering in the file.
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    carva_file_path, master_csv_path = paths["carva_csv"], paths["master_csv"]
    
    if not Path(master_csv_path).exists():
        print(f"❌ Master CSV file not found: {master_csv_path}.")
        return
        
    df_master = pd.read_csv(master_csv_path)

    def extract_overlapping_segments(arr, window_size, hop_factor):
        # ... function content ...
        segments, starts = [], []
        hop_size = max(1, int(window_size / hop_factor))
        i = 0
        while i <= len(arr) - window_size:
            segment = arr[i:i + window_size]
            if not np.any(np.isnan(segment)):
                segments.append(segment)
                starts.append(i)
            i += hop_size
        return segments, starts

    def process_one_song_pca(song_idx):
        # ... function content ...
        song_df = df_master[df_master["Index"] == song_idx].reset_index(drop=True)
        if song_df.empty: return []
        
        audio_path = song_df.loc[0, "AudioPath"]
        original_song_data = song_df["Tonic_Normalized_Frequency"].values
        remaining_data = original_song_data.copy()
        all_found_segments = []
        global_label_offset = 0
        window_size = config["initial_window_size"]
        
        while window_size >= config["min_window_size"]:
            segments, segment_starts = extract_overlapping_segments(remaining_data, window_size, config["hop_factor"])
            
            if len(segments) < config.get("outlier_threshold", 2):
                window_size -= config["decay_size"]
                continue

            X_abs = np.stack(segments)
            X_shape = X_abs - np.mean(X_abs, axis=1, keepdims=True)
            X_combined = np.concatenate([X_abs, X_shape], axis=1)
            pca = PCA(n_components=min(config["pca_components"], X_combined.shape[1]))
            X_pca = pca.fit_transform(X_combined)
            dist_matrix = squareform(pdist(X_pca, metric='euclidean'))
            
            if dist_matrix.size > 0:
                large_distance = np.max(dist_matrix) * 10 + 1 
                for i in range(len(segments)):
                    for j in range(i + 1, len(segments)):
                        start_i, start_j = segment_starts[i], segment_starts[j]
                        if start_j < (start_i + window_size) and start_i < (start_j + window_size):
                            dist_matrix[i, j] = large_distance
                            dist_matrix[j, i] = large_distance
            
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=config["similarity_threshold"], metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dist_matrix)
            
            cluster_origins = defaultdict(list)
            for seg_idx, start_pos in enumerate(segment_starts):
                cluster_origins[labels[seg_idx]].append(start_pos)
            
            for label, starts in cluster_origins.items():
                if len(starts) >= config["outlier_threshold"]:
                    global_label = global_label_offset + label
                    for start_frame in starts:
                        segment_data = original_song_data[start_frame : start_frame + window_size]
                        all_found_segments.append([start_frame, start_frame + window_size, global_label, segment_data])
                        remaining_data[start_frame : start_frame + window_size] = np.nan
            
            global_label_offset += (len(set(labels)) + 1)
            window_size -= config["decay_size"]
        
        rows = []
        old_labels = sorted(set(lbl for _, _, lbl, _ in all_found_segments))
        label_map  = {old: new for new, old in enumerate(old_labels)}
        
        for start, end, lbl, seg_data in all_found_segments:
            rows.append({
                "Index": int(song_idx), "AudioPath": audio_path,
                "SegmentList": json.dumps(list(seg_data)),
                "StartFrame": int(start), "EndFrame": int(end),
                "Label": label_map.get(lbl, -1)
            })
        return rows
    
    print(f"🔍 Starting optimized clustering for {raaga_name}...")
    if Path(carva_file_path).exists():
        print(f"🗑️ Deleting old carva file: {carva_file_path}")
        os.remove(carva_file_path)
    
    if song_index is None:
        song_indices = sorted(df_master["Index"].unique())
        results = Parallel(n_jobs=os.cpu_count(), backend="loky")(delayed(process_one_song_pca)(idx) for idx in tqdm(song_indices, desc="Processing songs"))
        flat_rows = [row for song_rows in results for row in song_rows]
    else:
        print(f"🎵 Processing a single song (Index: {song_index})...")
        flat_rows = process_one_song_pca(song_index)

    if not flat_rows:
        print("⚠️ No motifs found with the current parameters.")
        return pd.DataFrame()

    carva_df = pd.DataFrame(flat_rows)
    carva_df.to_csv(carva_file_path, index=False)
    print(f"✅ Clustering finished — Found {len(carva_df)} segments. Saved to {carva_file_path}")
    return carva_df

# ============================================================================
# PHASE 2: SECONDARY CLUSTERING
# ============================================================================

def recluster_with_dtw(audio_dir: str, config: Dict, song_index: Optional[int] = None):
    # ... function content ...
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    carva_path = paths["carva_csv"]
    print(f"🔬 Starting DTW re-clustering for '{raaga_name}'...")
    try: carva_df = pd.read_csv(carva_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {carva_path}"); return

    target_df = carva_df[carva_df['Index'] == song_index].copy() if song_index is not None else carva_df.copy()
    print(f" targeting song with Index: {song_index}" if song_index is not None else " targeting all songs.")

    if target_df.empty:
        print("⚠️ No segments found for the specified criteria."); return
        
    interpolation_size = config.get("interpolation_size", 50)
    similarity_threshold = config.get("dtw_similarity_threshold", 1.5)
    interpolated_segments, valid_original_indices = [], target_df.index.tolist()
    
    for seg_str in target_df['SegmentList']:
        try:
            segment = clean_np_float_list(seg_str)
            interpolated_segments.append(np.array(interpolate_list(segment, interpolation_size)) if len(segment) > 0 else None)
        except (ValueError, SyntaxError):
            interpolated_segments.append(None)
    
    final_segments = [seg for seg in interpolated_segments if seg is not None]
    final_indices = [idx for i, idx in enumerate(valid_original_indices) if interpolated_segments[i] is not None]

    print(f"Found {len(final_segments)} valid segments to re-cluster.")
    if len(final_segments) < 2:
        print("⚠️ Not enough valid segments (< 2) to perform DTW clustering."); return

    num_segments = len(final_segments)
    dist_matrix = np.zeros((num_segments, num_segments))
    for i in tqdm(range(num_segments), desc="Calculating DTW Matrix"):
        for j in range(i + 1, num_segments):
            dist, _ = fastdtw(final_segments[i], final_segments[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    print("🧬 Performing hierarchical clustering...")
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=similarity_threshold, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(dist_matrix)

    if 'DTW_Label' not in carva_df.columns: carva_df['DTW_Label'] = -1
    carva_df.loc[final_indices, 'DTW_Label'] = labels
    carva_df.to_csv(carva_path, index=False)
    
    print(f"\n✅ DTW re-clustering finished! Found {len(set(labels))} new clusters.")
    print(f"💾 Updated {carva_path} with 'DTW_Label' column.")

def recluster_with_pca(audio_dir: str, config: Dict, song_index: Optional[int] = None):
    # ... function content ...
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    carva_path = paths["carva_csv"]
    print(f"🔬 Starting PCA re-clustering for '{raaga_name}'...")
    try: carva_df = pd.read_csv(carva_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {carva_path}"); return

    target_df = carva_df[carva_df['Index'] == song_index].copy() if song_index is not None else carva_df.copy()
    print(f" targeting song with Index: {song_index}" if song_index is not None else " targeting all songs.")

    if target_df.empty:
        print("⚠️ No segments found for the specified criteria."); return
        
    interpolation_size = config.get("interpolation_size", 50)
    similarity_threshold = config.get("pca_similarity_threshold", 0.7)
    pca_components = config.get("pca_components", 10)
    interpolated_segments, valid_original_indices = [], target_df.index.tolist()
    
    for seg_str in target_df['SegmentList']:
        try:
            segment = clean_np_float_list(seg_str)
            interpolated_segments.append(np.array(interpolate_list(segment, interpolation_size)) if len(segment) > 0 else None)
        except (ValueError, SyntaxError):
            interpolated_segments.append(None)
    
    final_segments = [seg for seg in interpolated_segments if seg is not None]
    final_indices = [idx for i, idx in enumerate(valid_original_indices) if interpolated_segments[i] is not None]

    print(f"Found {len(final_segments)} valid segments to re-cluster.")
    if len(final_segments) < 2:
        print("⚠️ Not enough valid segments (< 2) to perform PCA clustering."); return

    print("🤖 Performing PCA transformation...")
    X_abs = np.stack(final_segments)
    X_shape = X_abs - np.mean(X_abs, axis=1, keepdims=True)
    X_combined = np.concatenate([X_abs, X_shape], axis=1)
    pca = PCA(n_components=min(pca_components, X_combined.shape[1]))
    X_pca = pca.fit_transform(X_combined)
    dist_matrix = squareform(pdist(X_pca, metric='euclidean'))
            
    print("🧬 Performing hierarchical clustering...")
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=similarity_threshold, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(dist_matrix)

    if 'PCA_Label' not in carva_df.columns: carva_df['PCA_Label'] = -1
    carva_df.loc[final_indices, 'PCA_Label'] = labels
    carva_df.to_csv(carva_path, index=False)
    
    print(f"\n✅ PCA re-clustering finished! Found {len(set(labels))} new clusters.")
    print(f"💾 Updated {carva_path} with 'PCA_Label' column.")

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def play_and_plot_cluster(audio_dir: str, cluster_number: int, song_index: int, sr: int = 44100):
    """
    Plots segments of a cluster from a specific song in context, then shows a second 
    plot with only those segments overlaid for similarity, and plays their audio.
    """
    raaga_name = Path(audio_dir).name
    if raaga_name.endswith('_Vocals'):
        raaga_name = raaga_name.replace('_Vocals', '')

    paths = get_data_paths(raaga_name)
    
    try:
        master_df = pd.read_csv(paths["master_csv"])
        carva_df = pd.read_csv(paths["carva_csv"])
    except FileNotFoundError as e:
        print(f"❌ Required data file not found: {e}")
        return

    label_col = 'Second Labels' if 'Second Labels' in carva_df.columns else 'Label'
    
    # Filter for segments from the specific cluster AND song
    cluster_in_song_segments = carva_df[
        (carva_df[label_col] == cluster_number) &
        (carva_df['Index'] == song_index)
    ]
    
    if cluster_in_song_segments.empty:
        print(f"⚠️ No segments found for cluster {cluster_number} in song index {song_index}.")
        return

    song_data = master_df[master_df["Index"] == song_index].reset_index(drop=True)
    if song_data.empty:
        print(f"⚠️ No data found for song index {song_index} in the master CSV.")
        return

    print(f"🔍 Analyzing Cluster {cluster_number} in Song Index {song_index}...")

    # PLOT 1: Segments in their original song context
    plt.style.use('dark_background')
    plt.figure(figsize=(18, 8))
    ax1 = plt.gca()

    full_frequency = song_data["Frequency"].values
    full_time = np.arange(len(full_frequency))
    tonic_note = song_data["Tonic"].iloc[0]
    song_name = song_data["SongName"].iloc[0]

    ax1.plot(full_time, full_frequency, color='gray', alpha=0.5, label='F0 Contour')
    
    random.seed(cluster_number)
    cluster_color = '#%06x' % random.randint(0, 0xFFFFFF)
    
    for i, (_, row) in enumerate(cluster_in_song_segments.iterrows()):
        start_frame = int(row['StartFrame'])
        end_frame = int(row['EndFrame'])
        ax1.plot(full_time[start_frame:end_frame], 
                 full_frequency[start_frame:end_frame], 
                 color=cluster_color, linewidth=2, 
                 label=f"Cluster {cluster_number}" if i == 0 else "")
            
    carnatic_frequencies = {note: librosa.note_to_hz(tonic_note) * ratio for note, ratio in CARNATIC_RATIOS.items()}
    valid_freqs = song_data['Frequency'].dropna()
    if not valid_freqs.empty:
        min_freq, max_freq = valid_freqs.min(), valid_freqs.max()
        for note, freq in carnatic_frequencies.items():
            if min_freq <= freq <= max_freq:
                ax1.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
                ax1.text(ax1.get_xlim()[1] * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
    
    ax1.set_title(f"Context Plot for Cluster {cluster_number} in Song: '{song_name}'")
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- MODIFIED PLOT 2 ---
    # PLOT 2: Segments from THIS SONG ONLY overlaid for similarity
    print(f"\n📈 Plotting the {len(cluster_in_song_segments)} segments from Cluster {cluster_number} in this song, overlaid for comparison...")
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 7))
    ax2 = plt.gca()
    
    all_freqs_in_cluster = []

    # Iterate over the segments from THIS SONG ONLY
    for _, row in cluster_in_song_segments.iterrows():
        seg_start = int(row['StartFrame'])
        seg_end = int(row['EndFrame'])
        
        # We can use the 'song_data' dataframe directly since we already filtered for this song
        segment_freq_data = song_data['Frequency'].values[seg_start:seg_end]
        
        if len(segment_freq_data) > 0:
            ax2.plot(np.arange(len(segment_freq_data)), segment_freq_data, color='cyan', alpha=0.4, linewidth=1.5)
            all_freqs_in_cluster.extend(segment_freq_data)
            
    if all_freqs_in_cluster:
        valid_cluster_freqs = [f for f in all_freqs_in_cluster if pd.notna(f)]
        if valid_cluster_freqs:
            min_freq_cluster, max_freq_cluster = min(valid_cluster_freqs), max(valid_cluster_freqs)
            for note, freq in carnatic_frequencies.items():
                if min_freq_cluster <= freq <= max_freq_cluster:
                    ax2.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
                    ax2.text(ax2.get_xlim()[1] * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
            ax2.set_ylim(min_freq_cluster - 10, max_freq_cluster + 10)

    ax2.set_title(f"Shape Comparison of Segments in Cluster {cluster_number} from Song '{song_name}'")
    ax2.set_xlabel("Time (frames within segment)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # AUDIO PLAYBACK (unchanged)
    print("\n🎧 Playing segments from the specified song:")
    for _, row in cluster_in_song_segments.iterrows():
        start_frame = int(row['StartFrame'])
        end_frame = int(row['EndFrame'])
        audio_path = row['AudioPath']
        
        print(f"   - Frames: {start_frame}-{end_frame}")
        song_time_data = song_data['Time'].reset_index(drop=True)
        start_time = song_time_data.get(start_frame, 0)
        end_time = song_time_data.get(end_frame, song_time_data.iloc[-1])
        
        try:
            audio, _ = librosa.load(audio_path, sr=sr)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if end_sample > start_sample:
                ipd.display(ipd.Audio(audio[start_sample:end_sample], rate=sr))
            else:
                print("   ⚠️ Invalid segment length, skipping playback.")
        except Exception as e:
            print(f"   ⚠️ Could not load or play audio from {audio_path}: {e}")

    print("---------------------------------")

def evaluate_clustering_results(audio_dir: str, song_idx: int):
    """
    Evaluates and visualizes clustering results for a specific song,
    automatically finding the required CSV files.
    """
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    
    try:
        master_df = pd.read_csv(paths["master_csv"])
        # --- FIX: Automatically find the carva.csv file ---
        carva_df = pd.read_csv(paths["carva_csv"])
    except FileNotFoundError as e:
        print(f"❌ Required data file not found: {e}."); return

    song_master_df = master_df[master_df["Index"] == song_idx]
    if song_master_df.empty:
        print(f"⚠️ No data found for song index {song_idx} in the master file."); return
        
    song_name = song_master_df["SongName"].iloc[0]
    print(f"📊 Evaluating clustering for song: '{song_name}' (Index: {song_idx})")
    total_song_frames = len(song_master_df)
    song_carva_df = carva_df[carva_df['Index'] == song_idx]

    # --- Calculation of percentage clustered ---
    if song_carva_df.empty:
        total_clustered_frames = 0
    else:
        is_clustered = np.zeros(total_song_frames, dtype=bool)
        for _, row in song_carva_df.iterrows():
            start, end = int(row['StartFrame']), int(row['EndFrame'])
            if end > start and end <= total_song_frames:
                is_clustered[start:end] = True
        total_clustered_frames = np.sum(is_clustered)

    percentage_clustered = (total_clustered_frames / total_song_frames) * 100 if total_song_frames > 0 else 0
    print(f"   • Percentage of song clustered: {percentage_clustered:.2f}%")

    # --- Calculation of cluster count ---
    num_clusters = song_carva_df['Label'].nunique() if not song_carva_df.empty else 0
    print(f"   • Total number of clusters found: {num_clusters}")

    # --- Plotting metrics ---
    if not song_carva_df.empty:
        song_carva_df['SegmentLength'] = song_carva_df['EndFrame'] - song_carva_df['StartFrame']
        
        # Plot 1: Histogram of segment lengths
        plt.style.use('dark_background'); plt.figure(figsize=(10, 6))
        cluster_lengths = song_carva_df.groupby('Label')['SegmentLength'].first()
        if not cluster_lengths.empty:
            plt.hist(cluster_lengths, bins=range(cluster_lengths.min(), cluster_lengths.max() + 2), edgecolor='white')
            plt.title(f"Distribution of Cluster Segment Lengths for '{song_name}'")
            plt.xlabel("Segment Length (frames)"), plt.ylabel("Number of Clusters"), plt.grid(alpha=0.3), plt.show()

        # Plot 2: Bar chart of cluster sizes
        plt.style.use('dark_background'); plt.figure(figsize=(12, 8))
        cluster_sizes = song_carva_df.groupby('Label')['Index'].count().sort_values(ascending=False)
        cluster_sizes.plot(kind='bar', edgecolor='white', alpha=0.8, color='cyan')
        plt.title(f"Number of Segments per Cluster for '{song_name}'")
        plt.xlabel("Cluster Label"), plt.ylabel("Number of Segments"), plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3), plt.tight_layout(), plt.show()
    else:
        print("   • No clustered segments to plot.")

def label_cluster(audio_dir: str, cluster_number: int, swara_label: str):
    # ... function content ...
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    carva_csv_path = paths["carva_csv"]
    if not os.path.exists(carva_csv_path):
        print(f"❌ File not found: {carva_csv_path}"); return

    df = pd.read_csv(carva_csv_path)
    if 'Swara' not in df.columns: df['Swara'] = ""
    label_col = 'Second Labels' if 'Second Labels' in df.columns else 'Label'
    mask = df[label_col] == cluster_number
    match_count = mask.sum()

    if match_count == 0:
        print(f"⚠️ No rows found for cluster {cluster_number}. Nothing updated.")
    else:
        df.loc[mask, 'Swara'] = swara_label
        df.to_csv(carva_csv_path, index=False)
        print(f"✅ Labeled {match_count} rows in cluster {cluster_number} as '{swara_label}'.")

def cluster_curve(audio_dir: str, song_index: int, clusters_to_plot: Optional[Union[int, List[int]]] = None):
    """
    Plots the F0 curve for a song, highlighting specified clusters without a legend.
    """
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)

    try:
        master_df = pd.read_csv(paths["master_csv"])
        carva_df = pd.read_csv(paths["carva_csv"])
    except FileNotFoundError as e:
        print(f"❌ Required data file not found: {e}"); return

    song_data = master_df[master_df["Index"] == song_index]
    if song_data.empty:
        print(f"⚠️ No data found for song with index {song_index}."); return

    clustered_segments = carva_df[carva_df["Index"] == song_index]

    if clusters_to_plot is not None:
        clusters_to_plot = [clusters_to_plot] if isinstance(clusters_to_plot, int) else clusters_to_plot
        label_col = next((col for col in ['DTW_Label', 'PCA_Label', 'Second Labels', 'Label'] if col in clustered_segments.columns), 'Label')
        clustered_segments = clustered_segments[clustered_segments[label_col].isin(clusters_to_plot)]

    full_frequency = song_data["Frequency"].values
    x_axis_time = song_data["Time"].values
    tonic_note, song_name = song_data["Tonic"].iloc[0], song_data["SongName"].iloc[0]
    
    label_to_color = {}
    if not clustered_segments.empty:
        label_col = next((col for col in ['DTW_Label', 'PCA_Label', 'Second Labels', 'Label'] if col in clustered_segments.columns), 'Label')
        unique_labels = sorted(clustered_segments[label_col].unique())
        cmap = plt.get_cmap('tab20', len(unique_labels))
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    plt.style.use('dark_background'); plt.figure(figsize=(18, 8))
    plt.plot(x_axis_time, full_frequency, color='gray', alpha=0.5)

    if not clustered_segments.empty:
        label_col = next((col for col in ['DTW_Label', 'PCA_Label', 'Second Labels', 'Label'] if col in clustered_segments.columns), 'Label')
        for _, row in clustered_segments.iterrows():
            start, end, label = int(row['StartFrame']), int(row['EndFrame']), row[label_col]
            color = label_to_color.get(label, 'white')
            # Plot without the 'label' argument
            plt.plot(x_axis_time[start:end], full_frequency[start:end], color=color, linewidth=2.5)

    carnatic_frequencies = {note: librosa.note_to_hz(tonic_note) * ratio for note, ratio in CARNATIC_RATIOS.items()}
    valid_freqs = song_data['Frequency'].dropna()
    if not valid_freqs.empty:
        min_freq, max_freq = valid_freqs.min(), valid_freqs.max()
        # Use plt.xlim() only after plotting to ensure the limits are set
        plot_end_time = plt.xlim()[1] 
        for note, freq in carnatic_frequencies.items():
            if min_freq <= freq <= max_freq:
                plt.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
                plt.text(plot_end_time * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
    
    plt.title(f"Clustered F0 Contour for Song: '{song_name}' (Tonic: {tonic_note})")
    plt.xlabel("Time (seconds)"), plt.ylabel("Frequency (Hz)")
    
    # --- LEGEND CODE REMOVED ---
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # ---------------------------
    
    plt.grid(alpha=0.3), plt.tight_layout()
    if not valid_freqs.empty: plt.ylim(valid_freqs.min() - 10, valid_freqs.max() + 10)
    plt.show()

# def play_and_plot_secondary_cluster(audio_dir: str, cluster_number: int, song_index: int, sr: int = 44100):
#     # ... function content ...
#     raaga_name = Path(audio_dir).name.replace('_Vocals', '')
#     paths = get_data_paths(raaga_name)
#     try:
#         master_df, carva_df = pd.read_csv(paths["master_csv"]), pd.read_csv(paths["carva_csv"])
#     except FileNotFoundError as e:
#         print(f"❌ Required data file not found: {e}"); return

#     label_col = next((col for col in ['DTW_Label', 'PCA_Label', 'Second Labels'] if col in carva_df.columns), None)
#     if not label_col:
#         print("❌ Error: No secondary label column found in carva.csv."); return
#     print(f"INFO: Using secondary cluster column: '{label_col}'")
    
#     cluster_in_song_segments = carva_df[(carva_df[label_col] == cluster_number) & (carva_df['Index'] == song_index)]
#     if cluster_in_song_segments.empty:
#         print(f"⚠️ No segments found for secondary cluster {cluster_number} in song index {song_index}."); return

#     song_data = master_df[master_df["Index"] == song_index].reset_index(drop=True)
#     if song_data.empty:
#         print(f"⚠️ No data found for song index {song_index} in the master CSV."); return

#     print(f"🔍 Analyzing Secondary Cluster {cluster_number} in Song Index {song_index}...")

#     # PLOT 1
#     plt.style.use('dark_background'); plt.figure(figsize=(18, 8)); ax1 = plt.gca()
#     full_frequency, x_axis_time = song_data["Frequency"].values, song_data["Time"].values
#     tonic_note, song_name = song_data["Tonic"].iloc[0], song_data["SongName"].iloc[0]
#     ax1.plot(x_axis_time, full_frequency, color='gray', alpha=0.5, label='F0 Contour')
#     random.seed(cluster_number); cluster_color = '#%06x' % random.randint(0, 0xFFFFFF)
#     for i, (_, row) in enumerate(cluster_in_song_segments.iterrows()):
#         start_frame, end_frame = int(row['StartFrame']), int(row['EndFrame'])
#         ax1.plot(x_axis_time[start_frame:end_frame], full_frequency[start_frame:end_frame], color=cluster_color, linewidth=2, label=f"Cluster {cluster_number}" if i == 0 else "")
#     carnatic_frequencies = {note: librosa.note_to_hz(tonic_note) * ratio for note, ratio in CARNATIC_RATIOS.items()}
#     valid_freqs = song_data['Frequency'].dropna()
#     if not valid_freqs.empty:
#         min_freq, max_freq = valid_freqs.min(), valid_freqs.max()
#         plot_end_time = ax1.get_xlim()[1]
#         for note, freq in carnatic_frequencies.items():
#             if min_freq <= freq <= max_freq:
#                 ax1.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
#                 ax1.text(plot_end_time * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
#     ax1.set_title(f"Context Plot for Secondary Cluster {cluster_number} in Song: '{song_name}'")
#     ax1.set_xlabel("Time (seconds)"), ax1.set_ylabel("Frequency (Hz)"), ax1.legend(loc='upper right'), ax1.grid(alpha=0.3)
#     plt.tight_layout(), plt.show()

#     # PLOT 2
#     print(f"\n📈 Plotting the {len(cluster_in_song_segments)} segments from Secondary Cluster {cluster_number} in this song, overlaid for comparison...")
#     plt.style.use('dark_background'), plt.figure(figsize=(12, 7)), (ax2 := plt.gca())
#     all_freqs_in_cluster = []
#     for _, row in cluster_in_song_segments.iterrows():
#         start_frame, end_frame = int(row['StartFrame']), int(row['EndFrame'])
#         segment_freq_data = song_data['Frequency'].values[start_frame:end_frame]
#         if len(segment_freq_data) > 0:
#             ax2.plot(np.arange(len(segment_freq_data)), segment_freq_data, color='cyan', alpha=0.4, linewidth=1.5)
#             all_freqs_in_cluster.extend(segment_freq_data)
#     if all_freqs_in_cluster:
#         valid_cluster_freqs = [f for f in all_freqs_in_cluster if pd.notna(f)]
#         if valid_cluster_freqs:
#             min_freq_cluster, max_freq_cluster = min(valid_cluster_freqs), max(valid_cluster_freqs)
#             for note, freq in carnatic_frequencies.items():
#                 if min_freq_cluster <= freq <= max_freq_cluster:
#                     ax2.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
#                     ax2.text(ax2.get_xlim()[1] * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
#             ax2.set_ylim(min_freq_cluster - 10, max_freq_cluster + 10)
#     ax2.set_title(f"Shape Comparison of Segments in Secondary Cluster {cluster_number} from Song '{song_name}'")
#     ax2.set_xlabel("Time (frames within segment)"), ax2.set_ylabel("Frequency (Hz)"), ax2.grid(alpha=0.3)
#     plt.tight_layout(), plt.show()

#     # AUDIO PLAYBACK
#     print("\n🎧 Playing segments from the specified song:")
#     song_time_data = song_data['Time'].values
#     audio_path = cluster_in_song_segments['AudioPath'].iloc[0]
#     try:
#         audio, native_sr = librosa.load(audio_path, sr=None)
#         if sr and native_sr != sr: audio = librosa.resample(y=audio, orig_sr=native_sr, target_sr=sr)
#     except Exception as e:
#         print(f"⚠️ Could not load audio from {audio_path}: {e}"); return
#     for _, row in cluster_in_song_segments.iterrows():
#         start_frame, end_frame = int(row['StartFrame']), int(row['EndFrame'])
#         if start_frame >= len(song_time_data) or end_frame > len(song_time_data):
#             print(f"   - Frames: {start_frame}-{end_frame}: ⚠️ Invalid frame indices, skipping."); continue
#         start_time, end_time = song_time_data[start_frame], song_time_data[end_frame - 1]
#         print(f"   - Frames: {start_frame}-{end_frame} -> Playing from {start_time:.2f}s to {end_time:.2f}s")
#         start_sample, end_sample = int(start_time * sr), int(end_time * sr)
#         if end_sample > start_sample:
#             ipd.display(ipd.Audio(audio[start_sample:end_sample], rate=sr))
#         else:
#             print("   ⚠️ Calculated segment duration is zero, skipping.")
#     print("---------------------------------")

def play_and_plot_secondary_cluster(audio_dir: str, cluster_number: int, song_index: Optional[int] = None, sr: int = 44100):
    """
    Analyzes and visualizes a secondary cluster.

    - If song_index is provided, it shows plots and plays audio for that single song.
    - If song_index is None, it finds all songs containing the cluster, creates plots
      for each one, and then plays back all segment audio from all songs.
    - It also prints the assigned Swara label for the cluster, if it exists.
    """
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    
    try:
        master_df, carva_df = pd.read_csv(paths["master_csv"]), pd.read_csv(paths["carva_csv"])
    except FileNotFoundError as e:
        print(f"❌ Required data file not found: {e}"); return

    # --- 1. Find the secondary label column and filter for the cluster ---
    label_col = next((col for col in ['DTW_Label', 'PCA_Label', 'Second Labels'] if col in carva_df.columns), None)
    if not label_col:
        print("❌ Error: No secondary label column found in carva.csv."); return
    
    print(f"INFO: Using secondary cluster column: '{label_col}'")
    
    cluster_df = carva_df[carva_df[label_col] == cluster_number]
    if cluster_df.empty:
        print(f"⚠️ No segments found for secondary cluster {cluster_number}."); return

    # --- 2. Determine which mode to run in (Single Song vs. All Songs) ---
    if song_index is not None:
        # --- SINGLE-SONG MODE ---
        song_ids_to_process = [song_index]
        print(f"\n🔍 Analyzing Secondary Cluster {cluster_number} in single song (Index: {song_index})...")
    else:
        # --- ALL-SONGS MODE ---
        song_ids_to_process = sorted(cluster_df['Index'].unique())
        print(f"\n🔍 Analyzing Secondary Cluster {cluster_number} across {len(song_ids_to_process)} songs...")

    # --- 3. Loop through songs and generate PLOTS ---
    for s_idx in song_ids_to_process:
        song_data = master_df[master_df["Index"] == s_idx].reset_index(drop=True)
        cluster_in_song_segments = cluster_df[cluster_df['Index'] == s_idx]

        if song_data.empty or cluster_in_song_segments.empty:
            print(f"\n--- Skipping Song {s_idx} (No data or no segments for this cluster) ---")
            continue
            
        song_name = song_data["SongName"].iloc[0]
        print(f"\n--- Displaying plots for Song: '{song_name}' (Index: {s_idx}) ---")

        # PLOT 1: Context Plot
        plt.style.use('dark_background'); plt.figure(figsize=(18, 8)); ax1 = plt.gca()
        full_frequency, x_axis_time = song_data["Frequency"].values, song_data["Time"].values
        tonic_note = song_data["Tonic"].iloc[0]
        ax1.plot(x_axis_time, full_frequency, color='gray', alpha=0.5, label='F0 Contour')
        random.seed(cluster_number); cluster_color = '#%06x' % random.randint(0, 0xFFFFFF)
        for i, (_, row) in enumerate(cluster_in_song_segments.iterrows()):
            start_frame, end_frame = int(row['StartFrame']), int(row['EndFrame'])
            ax1.plot(x_axis_time[start_frame:end_frame], full_frequency[start_frame:end_frame], color=cluster_color, linewidth=2, label=f"Cluster {cluster_number}" if i == 0 else "")
        
        # ... (Carnatic notes plotting logic) ...
        carnatic_frequencies = {note: librosa.note_to_hz(tonic_note) * ratio for note, ratio in CARNATIC_RATIOS.items()}
        valid_freqs = song_data['Frequency'].dropna()
        if not valid_freqs.empty:
            min_freq, max_freq = valid_freqs.min(), valid_freqs.max()
            plot_end_time = ax1.get_xlim()[1]
            for note, freq in carnatic_frequencies.items():
                if min_freq <= freq <= max_freq:
                    ax1.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
                    ax1.text(plot_end_time * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')

        ax1.set_title(f"Context Plot for Secondary Cluster {cluster_number} in Song: '{song_name}'")
        ax1.set_xlabel("Time (seconds)"), ax1.set_ylabel("Frequency (Hz)"), ax1.legend(loc='upper right'), ax1.grid(alpha=0.3)
        plt.tight_layout(), plt.show()

        # PLOT 2: Shape Comparison Plot
        print(f"📈 Plotting the {len(cluster_in_song_segments)} segments from this song, overlaid for comparison...")
        plt.style.use('dark_background'), plt.figure(figsize=(12, 7)), (ax2 := plt.gca())
        all_freqs_in_cluster = []
        for _, row in cluster_in_song_segments.iterrows():
            start_frame, end_frame = int(row['StartFrame']), int(row['EndFrame'])
            segment_freq_data = song_data['Frequency'].values[start_frame:end_frame]
            if len(segment_freq_data) > 0:
                ax2.plot(np.arange(len(segment_freq_data)), segment_freq_data, color='cyan', alpha=0.4, linewidth=1.5)
                all_freqs_in_cluster.extend(segment_freq_data)
        
        # ... (Carnatic notes plotting logic for plot 2) ...
        if all_freqs_in_cluster:
            valid_cluster_freqs = [f for f in all_freqs_in_cluster if pd.notna(f)]
            if valid_cluster_freqs:
                min_freq_cluster, max_freq_cluster = min(valid_cluster_freqs), max(valid_cluster_freqs)
                for note, freq in carnatic_frequencies.items():
                    if min_freq_cluster <= freq <= max_freq_cluster:
                        ax2.axhline(y=freq, color='orange', linestyle='--', linewidth=0.8)
                        ax2.text(ax2.get_xlim()[1] * 1.005, freq, note, color='orange', fontsize=9, verticalalignment='center')
                ax2.set_ylim(min_freq_cluster - 10, max_freq_cluster + 10)
        
        ax2.set_title(f"Shape Comparison of Segments in Secondary Cluster {cluster_number} from Song '{song_name}'")
        ax2.set_xlabel("Time (frames within segment)"), ax2.set_ylabel("Frequency (Hz)"), ax2.grid(alpha=0.3)
        plt.tight_layout(), plt.show()

    # --- 4. Play back ALL audio for the cluster ---
    print(f"\n---------------------------------")
    print(f"🎧 Playing all {len(cluster_df)} segments for Secondary Cluster {cluster_number}...")
    for audio_path, group in cluster_df.groupby('AudioPath'):
        print(f"\n--- Loading audio from: {Path(audio_path).name} ---")
        try:
            audio, native_sr = librosa.load(audio_path, sr=None)
            if sr and native_sr != sr: audio = librosa.resample(y=audio, orig_sr=native_sr, target_sr=sr)
        except Exception as e:
            print(f"⚠️ Could not load audio: {e}"); continue
        
        for _, row in group.iterrows():
            s_idx, start_frame, end_frame = row['Index'], int(row['StartFrame']), int(row['EndFrame'])
            song_time_data = master_df[master_df['Index'] == s_idx]['Time'].values
            if start_frame >= len(song_time_data) or end_frame > len(song_time_data):
                print(f"   - Frames {start_frame}-{end_frame}: ⚠️ Invalid frame indices, skipping."); continue
            start_time, end_time = song_time_data[start_frame], song_time_data[end_frame - 1]
            print(f"   - Song {s_idx}, Frames {start_frame}-{end_frame} -> Playing from {start_time:.2f}s to {end_time:.2f}s")
            start_sample, end_sample = int(start_time * sr), int(end_time * sr)
            if end_sample > start_sample:
                ipd.display(ipd.Audio(audio[start_sample:end_sample], rate=sr))

    # --- 5. Print the final Swara label ---
    if 'Swara' in cluster_df.columns and not cluster_df['Swara'].empty:
        swara_label = cluster_df['Swara'].iloc[0]
        if pd.notna(swara_label) and swara_label:
            print(f"\n---------------------------------")
            print(f"🎵 Assigned Swara Label for Cluster {cluster_number}: '{swara_label}'")
    print("=================================\n")

# ============================================================================
# MAIN EXECUTION SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Define the directory containing the audio files for the raaga
    audio_dir = "Mayamalavagowlai_Vocals"
    
    # Define the path to the carva CSV file (used for evaluation)
    raaga_name = Path(audio_dir).name.replace('_Vocals', '')
    paths = get_data_paths(raaga_name)
    carva_path = paths["carva_csv"]

    # --- Step 1: Run the initial processing and clustering ---
    # process_audio_directory(audio_dir)
    # add_normalized_frequency_column(audio_dir)
    
    # CONFIG = {
    #     "initial_window_size": 60, "decay_size": 2, "min_window_size": 20,
    #     "hop_factor": 12, "outlier_threshold": 3, "similarity_threshold": 0.5,
    #     "pca_components": 15
    # }
    # perform_multistage_clustering(audio_dir, CONFIG)

    # --- Step 2: Run a secondary clustering method ---
    # dtw_config = { "interpolation_size": 50, "dtw_similarity_threshold": 1.5 }
    # recluster_with_dtw(audio_dir, dtw_config)
    
    # pca_config = { "interpolation_size": 50, "pca_similarity_threshold": 0.5, "pca_components": 10 }
    # recluster_with_pca(audio_dir, pca_config, song_index=0)

    # --- Step 3: Analyze and evaluate the results ---
    # evaluate_clustering_results(audio_dir, 0, carva_path)
    play_and_plot_secondary_cluster(audio_dir, cluster_number=0, song_index=0)
    
