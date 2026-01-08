# train_asl_landmark_mlp.py
"""
Train an MLP classifier on hand landmark CSV data from asl_alphabet_train (including del, nothing, space).
Each row: label, x0, y0, ..., x20, y20
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

CSV_FILE = 'asl_landmark_data.csv'
EPOCHS = 30
BATCH_SIZE = 32

# === LOAD DATA ===
df = pd.read_csv(CSV_FILE)
labels = df['label'].values
X = df.drop('label', axis=1).values.astype('float32')

# Encode labels (handles all classes, including del, nothing, space)
lb = LabelBinarizer()
y = lb.fit_transform(labels)

# === TRAIN/VAL SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === MODEL ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val)
)

# === SAVE MODEL AND LABELS ===
model.save('asl_landmark_mlp_model.h5')
np.save('asl_landmark_labels.npy', lb.classes_)
print('Model and label classes saved!')
