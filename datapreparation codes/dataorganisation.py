import os
import pandas as pd

# --- DIRECTORY PATHS ---
audio_roots = {
    "control": "/Users/marimanish/Desktop/dd/control",
    "dementia": "/Users/marimanish/Desktop/dd/Dementia" # Capital 'D'
}

transcript_roots = {
    "control": "/Users/marimanish/Desktop/dd/pitt/control",
    "dementia": "/Users/marimanish/Desktop/dd/pitt/dementia" # lowercase 'd'
}

tasks = ['cookie', 'fluency', 'recall', 'sentence']
data_rows = []

def find_files_recursively(root_dir, extensions):
    """Crawl all subfolders to find audio or transcript files."""
    matches = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if file matches any of the extensions (mp3, wav, cha)
            if any(file.lower().endswith(ext) for ext in extensions):
                file_id = os.path.splitext(file)[0]
                # If we find multiple versions (e.g., mp3 and wav), we keep the first one found
                if file_id not in matches:
                    matches[file_id] = os.path.join(root, file)
    return matches

print("🚀 Starting Recursive Search...")

for label_name, audio_root in audio_roots.items():
    label_val = 0 if label_name == "control" else 1
    print(f"\nProcessing {label_name.upper()}...")
    
    for task in tasks:
        audio_task_dir = os.path.join(audio_root, task)
        trans_task_dir = os.path.join(transcript_roots[label_name], task)
        
        if not os.path.exists(audio_task_dir):
            print(f"  ⚠️ Task folder not found: {audio_task_dir}")
            continue

        # 1. Find all Audio (mp3/wav) recursively (dives into 0wav folders)
        audio_files = find_files_recursively(audio_task_dir, [".mp3", ".wav"])
        
        # 2. Find all Transcripts (.cha) recursively
        trans_files = find_files_recursively(trans_task_dir, [".cha"])
        
        task_count = 0
        for file_id, audio_path in audio_files.items():
            if file_id in trans_files:
                data_rows.append({
                    'participant_id': file_id,
                    'task': task,
                    'audio_path': audio_path,
                    'transcript_path': trans_files[file_id],
                    'label': label_val
                })
                task_count += 1
        
        print(f"  - {task}: Found {task_count} matched samples.")

# Create the final CSV
df = pd.DataFrame(data_rows)
df.to_csv("master_dataset_map.csv", index=False)

print("\n" + "="*40)
print(f"✅ FINAL TOTAL: {len(df)} samples matched.")
print(f"Control: {len(df[df['label']==0])} | Dementia: {len(df[df['label']==1])}")
print("New 'master_dataset_map.csv' is ready!")
print("="*40)