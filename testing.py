import os
import pandas as pd
import torchaudio
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Path to the trained model
model_path = "multi_output_voice_model.h5"
model = load_model(model_path)

# Path to the new directory with testing audio files
test_audio_dir = r"C:\Users\Mateusz\Downloads\archive\cv-valid-test\cv-valid-test"
output_csv_path = "predictions_output.csv"

# Function to preprocess audio files
def preprocess_audio(file_path, target_sr=16000, n_mfcc=13):
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=target_sr, n_mfcc=n_mfcc)
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0).T.numpy()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process test data
test_audio_files = [os.path.join(test_audio_dir, f) for f in os.listdir(test_audio_dir) if f.lower().endswith(".mp3")]
X_test, file_names = [], []
for file_path in test_audio_files:
    mfcc = preprocess_audio(file_path)
    if mfcc is not None and mfcc.size > 0:
        X_test.append(mfcc)
        file_names.append(os.path.basename(file_path))

if X_test:
    # Pad sequences
    expected_length = 4495
    X_test_padded = pad_sequences(X_test, maxlen=expected_length, dtype="float32", padding="post", truncating="post")

    # Predictions
    predictions = model.predict(X_test_padded)
    gender_preds = predictions["gender_output"]
    age_preds = predictions["age_output"]

    # Save results
    gender_map = {0: "male", 1: "female"}
    age_map = {0: "twenties", 1: "thirties", 2: "forties", 3: "fifties", 4: "sixties", 5: "seventies"}
    output_data = []
    for file_name, gender_prob, age_prob in zip(file_names, gender_preds, age_preds):
        gender = gender_map[np.argmax(gender_prob)]
        age = age_map[np.argmax(age_prob)]
        output_data.append({
            "filename": file_name,
            "predicted_gender": gender,
            "predicted_gender_probability": np.max(gender_prob),
            "predicted_age": age,
            "predicted_age_probability": np.max(age_prob)
        })
    pd.DataFrame(output_data).to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Training Data Analysis
training_csv_path = r"C:\Users\Mateusz\Downloads\archive\cv-valid-train.csv"
columns = ["filename", "text", "up_votes", "down_votes", "age", "gender", "accent", "duration"]
data = pd.read_csv(training_csv_path, delimiter=";", names=columns, header=0)
data = data.dropna(subset=["gender", "age"])
data["gender"] = data["gender"].map({"male": 0, "female": 1})
data["age"] = data["age"].map({"twenties": 0, "thirties": 1, "forties": 2, "fifties": 3, "sixties": 4, "seventies": 5})

# Remove invalid data
data = data[data["gender"].isin([0, 1])]  # Keep valid gender labels
data = data[data["age"].isin([0, 1, 2, 3, 4, 5])]  # Keep valid age labels

# Debugging: Check unique labels
print("Unique gender labels in data:", data["gender"].unique())
print("Unique age labels in data:", data["age"].unique())

# Compute class weights dynamically
gender_weights = compute_class_weight(
    "balanced",
    classes=np.array([0, 1]),  # Gender classes (fixed)
    y=data["gender"]
)

age_classes = np.unique(data["age"])  # Dynamically determine age classes
age_weights = compute_class_weight(
    "balanced",
    classes=age_classes,
    y=data["age"]
)

# Debugging: Print computed weights
print("Computed Gender Class Weights:", dict(enumerate(gender_weights)))
print("Computed Age Class Weights:", dict(zip(age_classes, age_weights)))
