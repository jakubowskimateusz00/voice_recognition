import os
import numpy as np
import pandas as pd
import torchaudio
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths
model_path = "enhanced_gender_detection_model.h5"
test_audio_dir = r"PATH_TO_TEST_AUDIO"
output_excel_path = "gender_predictions_updated.xlsx"

# Load the Model
model = load_model(model_path)

# Function to Extract MFCC Features
def extract_mfcc(file_path, n_mfcc=13, target_sr=16000):
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=target_sr, n_mfcc=n_mfcc)
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0).T.numpy()  # Transpose to match (time, features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process Test Files
test_audio_files = [os.path.join(test_audio_dir, f) for f in os.listdir(test_audio_dir) if f.lower().endswith(".mp3")]
X_test, file_names = [], []
for file_path in test_audio_files:
    mfcc = extract_mfcc(file_path)
    if mfcc is not None and mfcc.size > 0:
        X_test.append(mfcc)
        file_names.append(os.path.basename(file_path))

if X_test:
    # Pad Sequences
    max_seq_length = 4495
    X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_seq_length, padding="post", dtype="float32")

    # Predictions
    predictions = model.predict(X_test_padded)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_genders = ["male" if label == 0 else "female" for label in predicted_labels]

    # Calculate Confidence Scores
    confidence_scores = [np.max(pred) for pred in predictions]

    # Create DataFrame and Save
    results_df = pd.DataFrame({
        "File Name": file_names,
        "Predicted Gender": predicted_genders,
        "Prediction Confidence": confidence_scores
    })

    results_df.to_excel(output_excel_path, index=False)
    print(f"Predictions saved to {output_excel_path}")
else:
    print("No valid audio files found.")
