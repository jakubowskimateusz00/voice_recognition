import os
import pandas as pd
import torchaudio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.utils import to_categorical

# Load the dataset
columns = ["filename", "text", "up_votes", "down_votes", "age", "gender", "accent", "duration"]
csv_path = r"C:\Users\Mateusz\Downloads\archive\cv-valid-train.csv"
data = pd.read_csv(csv_path, delimiter=";", names=columns, header=0)

# Drop rows with missing or invalid gender or age
data = data.dropna(subset=["filename", "gender", "age"])
valid_genders = {"male": 0, "female": 1}
valid_ages = {"twenties": 0, "thirties": 1, "forties": 2, "fifties": 3, "sixties": 4, "seventies": 5}

# Map gender and age to integers
data["gender"] = data["gender"].map(valid_genders)
data["age"] = data["age"].map(valid_ages)

# Drop rows with unmapped gender or age values
data = data.dropna(subset=["gender", "age"])

# Extract filenames, gender, and age
file_paths = data["filename"].values
gender_labels = data["gender"].astype(int).values
age_labels = data["age"].astype(int).values

# Function to preprocess audio files
def preprocess_audio(file_path, target_sr=16000, n_mfcc=13):
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        mfcc_transform = torchaudio.transforms.MFCC(sample_rate=target_sr, n_mfcc=n_mfcc)
        mfcc = mfcc_transform(waveform)
        print(f"Processed file: {file_path}")
        return mfcc.squeeze(0).T.numpy()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Preprocess audio files
X, y_gender, y_age = [], [], []
for file_path, gender, age in zip(file_paths, gender_labels, age_labels):
    mfcc = preprocess_audio(file_path)
    if mfcc is not None and mfcc.size > 0:
        X.append(mfcc)
        y_gender.append(gender)
        y_age.append(age)

# Ensure there's data to process
if len(X) == 0:
    raise ValueError("No valid audio data was processed. Please check your audio files and paths.")

# Pad sequences for uniform input size
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, dtype="float32")
y_gender = to_categorical(y_gender, num_classes=2)  # One-hot encode gender (2 classes)
y_age = to_categorical(y_age, num_classes=6)  # One-hot encode age (6 classes)

# Debugging: Print target shapes
print(f"X_padded shape: {X_padded.shape}")
print(f"y_gender shape: {y_gender.shape}")
print(f"y_age shape: {y_age.shape}")

# Split data into training and testing sets
X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    X_padded, y_gender, y_age, test_size=0.2, random_state=42
)

# Debugging: Print target shapes before training
print(f"y_gender_train shape: {y_gender_train.shape}")
print(f"y_age_train shape: {y_age_train.shape}")

# Build the multi-output model
input_layer = Input(shape=(X_padded.shape[1], X_padded.shape[2]))
x = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64))(x)

# Output for gender prediction
gender_output = Dense(2, activation="softmax", name="gender_output")(x)

# Output for age prediction
age_output = Dense(6, activation="softmax", name="age_output")(x)

# Compile the model with explicit output mapping
model = Model(inputs=input_layer, outputs={"gender_output": gender_output, "age_output": age_output})
model.compile(
    optimizer="adam",
    loss={"gender_output": "categorical_crossentropy", "age_output": "categorical_crossentropy"},
    metrics={"gender_output": "accuracy", "age_output": "accuracy"}
)

# Debugging: Print model summary and output names
model.summary()
print("Model outputs:", model.output_names)

# Debugging: Verify target mappings
print(f"Target for gender_output: {y_gender_train.shape}")
print(f"Target for age_output: {y_age_train.shape}")

# Train the model
model.fit(
    X_train,
    {"gender_output": y_gender_train, "age_output": y_age_train},  # Explicit mapping
    validation_data=(X_test, {"gender_output": y_gender_test, "age_output": y_age_test}),
    epochs=10,
    batch_size=32
)

# Save the model
model.save("multi_output_voice_model.h5")
print("Model saved as multi_output_voice_model.h5")
