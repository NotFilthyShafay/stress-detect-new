import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

def preprocess_dataset(csv_path, output_dir='processed_data', test_size=0.2, random_state=42):
    """
    Preprocess the dataset by:
    1. Loading the CSV
    2. Handling any missing values
    3. Normalizing features
    4. Splitting into train/test sets (80-20 split)
    5. Saving the processed datasets
    
    Args:
        csv_path (str): Path to the CSV file
        output_dir (str): Directory to save processed data
        test_size (float): Proportion of data for test set (default: 0.2)
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Paths to the saved processed datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check for missing values and handle them
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values. Filling with appropriate values...")
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # If 'video_file' column exists, remove it from features
    if 'video_file' in X.columns:
        X = X.drop('video_file', axis=1)
    if 'fidget_percentage_fidget' in X.columns:
        X = X.drop('fidget_percentage_fidget', axis=1)
    if 'hand_movement_fidget' in X.columns:
        X = X.drop('hand_movement_fidget', axis=1)
    if 'face_movement_fidget' in X.columns:
        X = X.drop('face_movement_fidget', axis=1)
    if 'arm_movement_fidget' in X.columns:
        X = X.drop('arm_movement_fidget', axis=1)
    
    
    # Save the feature order for inference
    with open(os.path.join(output_dir, 'feature_order.json'), 'w') as f:
        json.dump(X.columns.tolist(), f)
        
    # Normalize/standardize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Split into train and test sets (80% train, 20% test)
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save column names for reference
    feature_names = X.columns.tolist()
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Save processed datasets
    print("Saving processed datasets...")
    train_data = np.column_stack((X_train, y_train.values.reshape(-1, 1)))
    test_data = np.column_stack((X_test, y_test.values.reshape(-1, 1)))
    
    train_path = os.path.join(output_dir, 'train_data.npy')
    test_path = os.path.join(output_dir, 'test_data.npy')
    
    np.save(train_path, train_data)
    np.save(test_path, test_data)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Preprocessing complete. Files saved to {output_dir}")
    
    return {
        'train_path': train_path,
        'test_path': test_path,
        'scaler_path': os.path.join(output_dir, 'scaler.pkl'),
        'feature_names_path': os.path.join(output_dir, 'feature_names.txt')
    }

if __name__ == "__main__":
    # Example usage
    preprocess_dataset(r'MAS\dataset_clean.csv')