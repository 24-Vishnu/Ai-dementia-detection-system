import pandas as pd
import librosa
import numpy as np
import pylangacq as pla
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_comprehensive_features(audio_path, transcript_path, participant_id, task, label):
    """
    Extract comprehensive features for dementia detection.
    Returns dictionary with ~86 acoustic + linguistic features.
    """
    features = {
        'participant_id': participant_id,
        'task': task,
        'label': label
    }
    
    try:
        # ===== LOAD AUDIO =====
        y, sr = librosa.load(audio_path, sr=16000, duration=60.0)
        
        # ===== 1. MFCC FEATURES (13 coefficients × 4 statistics = 52 features) =====
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Mean, Std, Max, Min for each MFCC
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
        
        # ===== 2. PAUSE DETECTION (CRITICAL FOR DEMENTIA!) =====
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = np.percentile(rms, 20)
        silent_frames = rms < silence_threshold
        
        # Calculate pause statistics
        pause_durations = []
        current_pause = 0
        for is_silent in silent_frames:
            if is_silent:
                current_pause += hop_length / sr
            else:
                if current_pause > 0.1:  # Pauses > 100ms
                    pause_durations.append(current_pause)
                current_pause = 0
        
        features['num_pauses'] = len(pause_durations)
        features['mean_pause_duration'] = np.mean(pause_durations) if pause_durations else 0
        features['max_pause_duration'] = np.max(pause_durations) if pause_durations else 0
        features['total_pause_time'] = np.sum(pause_durations) if pause_durations else 0
        features['pause_ratio'] = np.sum(silent_frames) / len(silent_frames)
        
        # ===== 3. SPEECH RATE =====
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        duration = len(y) / sr
        features['speech_rate'] = len(onset_frames) / duration if duration > 0 else 0
        features['total_duration'] = duration
        features['num_segments'] = len(onset_frames)
        
        # ===== 4. PITCH FEATURES =====
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variation'] = np.std(pitch_values) / np.mean(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            features['pitch_variation'] = 0
        
        # ===== 5. ENERGY FEATURES =====
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)
        features['energy_min'] = np.min(rms)
        
        # ===== 6. SPECTRAL FEATURES =====
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # ===== 7. PROSODIC FEATURES =====
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['num_beats'] = len(beats)
        
        # ===== 8. VOICE QUALITY =====
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        
        features['harmonic_noise_ratio'] = harmonic_energy / (percussive_energy + 1e-10)
        features['harmonic_energy'] = harmonic_energy
        features['percussive_energy'] = percussive_energy
        
        # ===== 9. LINGUISTIC FEATURES =====
        chat = pla.read_chat(transcript_path)
        par_words = chat.words(participants='PAR')
        par_utterances = chat.utterances(participants='PAR')
        
        features['word_count'] = len(par_words)
        features['utterance_count'] = len(par_utterances)
        
        # Type-Token Ratio (vocabulary diversity)
        if len(par_words) > 0:
            features['ttr'] = len(set(par_words)) / len(par_words)
            features['unique_words'] = len(set(par_words))
        else:
            features['ttr'] = 0
            features['unique_words'] = 0
        
        # Average words per utterance
        if len(par_utterances) > 0:
            features['avg_words_per_utterance'] = len(par_words) / len(par_utterances)
        else:
            features['avg_words_per_utterance'] = 0
        
        # Repetition detection
        if len(par_words) > 0:
            word_counts = {}
            for word in par_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            features['repetition_ratio'] = repeated_words / len(set(par_words))
        else:
            features['repetition_ratio'] = 0
        
        return features
        
    except Exception as e:
        print(f"❌ Error processing {participant_id}: {e}")
        return None


def run_extraction():
    """Main extraction pipeline"""
    
    # ═══════════════════════════════════════════════════════════════
    # 📝 CONFIGURATION: UPDATE THIS PATH TO YOUR CSV FILE
    # ═══════════════════════════════════════════════════════════════
    
    # Option 1: If CSV is in the same folder as extraction.py
    CSV_FILE_PATH = "/Users/marimanish/Desktop/dd/master_dataset_map.csv"
    
    # Option 2: If CSV is in a different folder (UNCOMMENT AND UPDATE)
    # CSV_FILE_PATH = "/Users/marimanish/Desktop/dd/master_dataset_map.csv"
    
    # Option 3: For Windows users (use raw string or double backslashes)
    # CSV_FILE_PATH = r"C:\Users\YourName\Desktop\dd\master_dataset_map.csv"
    
    # ═══════════════════════════════════════════════════════════════
    
    # Check if file exists
    if not os.path.exists(CSV_FILE_PATH):
        print("="*70)
        print("❌ ERROR: CSV file not found!")
        print("="*70)
        print(f"Looking for: {CSV_FILE_PATH}")
        print(f"Current directory: {os.getcwd()}")
        print("\n📝 Please update CSV_FILE_PATH in the code to match your file location.")
        print("\nExample locations:")
        print("  - Same folder: 'master_dataset_map.csv'")
        print("  - Full path (Mac): '/Users/yourname/Desktop/dd/master_dataset_map.csv'")
        print("  - Full path (Windows): r'C:\\Users\\yourname\\Desktop\\dd\\master_dataset_map.csv'")
        print("="*70)
        return
    
    print("="*70)
    print("DEMENTIA DETECTION - ENHANCED FEATURE EXTRACTION")
    print("="*70)
    
    # Load the CSV
    print(f"\n📂 Loading dataset from: {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total files: {len(df)}")
    print(f"   Dementia patients: {sum(df['label'] == 1)}")
    print(f"   Control (healthy): {sum(df['label'] == 0)}")
    print(f"\n   Columns: {list(df.columns)}")
    
    # Verify required columns
    required_columns = ['participant_id', 'task', 'audio_path', 'transcript_path', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n❌ ERROR: Missing required columns: {missing_columns}")
        print(f"   Found columns: {list(df.columns)}")
        return
    
    # Verify some audio files exist
    print(f"\n🔍 Verifying file paths...")
    sample_audio = df.iloc[0]['audio_path']
    sample_transcript = df.iloc[0]['transcript_path']
    
    print(f"   Sample audio path: {sample_audio}")
    print(f"   Sample transcript path: {sample_transcript}")
    
    if not os.path.exists(sample_audio):
        print(f"\n⚠️ WARNING: Audio file not found: {sample_audio}")
        print(f"   This might mean your audio_path column contains incorrect paths.")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"\n   💡 TIP: Make sure audio_path and transcript_path columns contain")
        print(f"          either absolute paths or paths relative to this script's location.")
    else:
        print(f"   ✓ Sample audio file found!")
    
    if not os.path.exists(sample_transcript):
        print(f"\n⚠️ WARNING: Transcript file not found: {sample_transcript}")
    else:
        print(f"   ✓ Sample transcript file found!")
    
    # Start extraction
    features_list = []
    failed_count = 0
    
    print("\n" + "="*70)
    print("🚀 Starting feature extraction...")
    print("   This will take approximately 10-20 minutes for 1241 files.")
    print("="*70 + "\n")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="file"):
        result = extract_comprehensive_features(
            audio_path=row['audio_path'],
            transcript_path=row['transcript_path'],
            participant_id=row['participant_id'],
            task=row['task'],
            label=row['label']
        )
        
        if result is not None:
            features_list.append(result)
        else:
            failed_count += 1
    
    # Create DataFrame
    if len(features_list) == 0:
        print("\n❌ ERROR: No features were extracted!")
        print("   Please check your audio and transcript file paths.")
        return
    
    features_df = pd.DataFrame(features_list)
    
    # Save results
    output_file = "processed_features_enhanced.csv"
    features_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE!")
    print("="*70)
    print(f"✓ Successfully processed: {len(features_df)} samples")
    print(f"✗ Failed: {failed_count} samples")
    print(f"✓ Total features extracted: {len(features_df.columns) - 3}")  # -3 for metadata
    print(f"✓ Output saved to: {output_file}")
    print("="*70)
    
    # Show feature breakdown
    print("\n📊 Feature Breakdown:")
    print(f"   - MFCC features: 52 (13 coefficients × 4 statistics)")
    print(f"   - Pause features: 5 (count, duration, ratio)")
    print(f"   - Speech rate: 3 (rate, duration, segments)")
    print(f"   - Pitch features: 4 (mean, std, range, variation)")
    print(f"   - Energy features: 4 (mean, std, max, min)")
    print(f"   - Spectral features: 6 (centroid, rolloff, bandwidth, ZCR)")
    print(f"   - Prosodic features: 2 (tempo, beats)")
    print(f"   - Voice quality: 3 (harmonic-noise ratio, energies)")
    print(f"   - Linguistic features: 7 (word count, TTR, repetitions)")
    print(f"   TOTAL: ~86 features")
    
    print("\n📈 Class Distribution in Output:")
    print(features_df['label'].value_counts())
    
    # Show sample of extracted features
    print("\n📋 Sample of Extracted Features (first 3 rows):")
    print(features_df.head(3).to_string())
    
    print("\n" + "="*70)
    print("✅ Next step: Run training.py to train the model")
    print("="*70)
    
    return features_df


if __name__ == '__main__':
    run_extraction()