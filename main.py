# Model Training Code
import os
import pandas as pd
import torchaudio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Paths
training_csv_path = r"C:\Users\Mateusz\Downloads\archive\cv-valid-train.csv"
audio_dir = r"C:\Users\Mateusz\Downloads\archive\cv-valid-train\cv-valid-train"
model_save_path = "enhanced_gender_detection_model.h5"

# Preprocessing Function
def extract_mfcc(file_path, target_sr=16000, n_mfcc=13):
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

# Load Training Data
columns = ["filename", "text", "up_votes", "down_votes", "age", "gender", "accent", "duration"]
data = pd.read_csv(training_csv_path, delimiter=";", names=columns, header=0)
data = data.dropna(subset=["gender"])
data["gender"] = data["gender"].map({"male": 0, "female": 1})
data = data[data["gender"].isin([0, 1])]

# Prepare Training Data
X_train, y_gender = [], []
for file_path, gender in zip(data["filename"], data["gender"]):
    file_path_full = os.path.join(audio_dir, file_path)
    mfcc = extract_mfcc(file_path_full)
    if mfcc is not None and mfcc.size > 0:
        X_train.append(mfcc)
        y_gender.append(gender)

if len(X_train) == 0:
    raise ValueError("No valid training data available.")

# Pad Sequences
max_sequence_length = 4495
X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length, dtype="float32", padding="post")
y_gender = tf.keras.utils.to_categorical(y_gender, num_classes=2)

# Train-Validation Split
X_train, X_val, y_gender_train, y_gender_val = train_test_split(X_train_padded, y_gender, test_size=0.2, random_state=42)

# Model Definition
input_layer = Input(shape=(max_sequence_length, 13))
x = Conv1D(64, kernel_size=5, activation="relu")(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=5, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Dropout(0.4)(x)
x = Bidirectional(LSTM(128))(x)
x = Dropout(0.4)(x)
output = Dense(2, activation="softmax", name="gender_output")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the Model
model.fit(X_train, y_gender_train, validation_data=(X_val, y_gender_val), epochs=20, batch_size=32)

# Save the Model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")