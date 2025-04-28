import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

# Load annotations and existing dataset
with open('MAS/annotations.json', 'r') as f:
    annotations = json.load(f)
    
df_clean = pd.read_csv('MAS/dataset_clean.csv')

# Print a sample of filenames from both sources to help with debugging
print("Annotation filenames (first 5):")
for i, filename in enumerate(list(annotations.keys())[:5]):
    print(f"  {filename}")

print("\nDataset filenames (first 5):")
for i, filename in enumerate(df_clean['video_file'].tolist()[:5]):
    print(f"  {filename}")

# Function to find the best match for a filename
def find_matching_row(video_file, df):
    # First try exact match
    matching_rows = df[df['video_file'] == video_file]
    if len(matching_rows) > 0:
        return matching_rows.iloc[0]
    
    # If no exact match, try matching the timestamp part
    try:
        # Extract timestamps (assuming format like "143.24_150.46.mp4")
        timestamps = re.findall(r'\d+\.\d+', video_file)
        if len(timestamps) >= 1:
            # Look for rows with any of these timestamps
            for timestamp in timestamps:
                matching_rows = df[df['video_file'].str.contains(timestamp)]
                if len(matching_rows) > 0:
                    print(f"Found partial match for {video_file}: {matching_rows.iloc[0]['video_file']}")
                    return matching_rows.iloc[0]
    except Exception as e:
        print(f"Error while matching {video_file}: {e}")
    
    return None

# Initialize dataset
fusion_dataset = []

# Process each video in annotations
for video_file, stress_labels in annotations.items():
    # Find matching row in dataset_clean.csv
    row_data = find_matching_row(video_file, df_clean)
    
    if row_data is None:
        print(f"Warning: No matching data found for {video_file}")
        continue
    
    # Determine truths from annotations
    has_stress = any('stress' in label and 'no stress' not in label for label in stress_labels)
    audio_stress = any('audio stress' in label for label in stress_labels)
    face_stress = any('face stress' in label for label in stress_labels)
    body_stress = any('body stress' in label for label in stress_labels)
    prosodic_stress = any('prosodic stress' in label for label in stress_labels)
    
    # Create dictionaries from existing features
    audio_features = {
        'sadness': float(row_data['sadness_audio_audio']),
        'joy': float(row_data['joy_audio_audio']),
        'love': float(row_data['love_audio_audio']),
        'anger': float(row_data['anger_audio_audio']),
        'fear': float(row_data['fear_audio_audio']),
        'surprise': float(row_data['surprise_audio_audio']),
        'speech_rate': float(row_data['speech_rate_audio_audio'])
    }
    
    face_features = {
        'angry': float(row_data['angry_face']),
        'disgust': float(row_data['disgust_face']),
        'fear': float(row_data['fear_face']),
        'happy': float(row_data['happy_face']),
        'sad': float(row_data['sad_face']),
        'surprise': float(row_data['surprise_face']),
        'neutral': float(row_data['neutral_face']),
        'face_offscreen_ratio': float(row_data['face_offscreen_ratio_face'])
    }
    
    prosodic_features = {
        'neutral': float(row_data['neu_prosodic_prosodic']),
        'happy': float(row_data['hap_prosodic_prosodic']),
        'sad': float(row_data['sad_prosodic_prosodic']),
        'angry': float(row_data['ang_prosodic_prosodic'])
    }
    
    fidget_features = {
        'movement_score': int(row_data['movement_score_fidget']),
        'fidget_percentage': float(row_data['fidget_percentage_fidget']),
        'hand_movement': float(row_data['hand_movement_fidget']),
        'face_movement': float(row_data['face_movement_fidget']),
        'arm_movement': float(row_data['arm_movement_fidget']),
        'overall_intensity': float(row_data['overall_intensity_fidget'])
    }
    
    # Create fusion dataset row
    fusion_row = {
        'video_file': video_file,
        'ground truth': int(has_stress),
        'audio_pred': str(audio_features),
        'face_pred': str(face_features),
        'fidget_pred': str(fidget_features),
        'prosodic_pred': str(prosodic_features),
        'audio_truth': int(audio_stress),
        'face_truth': int(face_stress),
        'fidget_truth': int(body_stress),
        'prosodic_truth': int(prosodic_stress)
    }
    
    fusion_dataset.append(fusion_row)
# Create DataFrame
fusion_df = pd.DataFrame(fusion_dataset)

#printds.head

print(f"\nCreated dataset with {len(fusion_df)} entries out of {len(annotations)} annotations")

# Split into train and validation sets
train_df, val_df = train_test_split(fusion_df, test_size=0.3, random_state=42)

# Save to CSV
train_df.to_csv('train_dataframe.csv', index=False)
val_df.to_csv('val_dataframe.csv', index=False)

print(f"Training set: {len(train_df)} entries")
print(f"Validation set: {len(val_df)} entries")