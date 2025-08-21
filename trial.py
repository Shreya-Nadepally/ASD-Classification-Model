import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
    TimeDistributed, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# === Dummy Dataset Setup ===
# Simulate 800 samples of 8x8 image data
X_img = np.random.rand(800, 8, 8, 1)

# Simulate labels (0 or 1) and one-hot encode
y = np.random.randint(0, 2, 800)
y_cat = to_categorical(y)

# === Train-test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_img, y_cat, test_size=0.2, random_state=42)

# === CNN Model ===
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

cnn_eval = cnn_model.evaluate(X_test, y_test)
print("CNN Accuracy:", cnn_eval[1])

# === T-CNN Model ===
# Simulate sequential data (e.g., splitting image into 4 sequences of 8x8 parts)
X_seq = X_img.reshape((800, 4, 8, 8, 1))

tcnn_model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(4, 8, 8, 1)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    GlobalAveragePooling1D(),  # Fix: remove time-step dimension
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

tcnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tcnn_model.fit(X_seq, y_cat, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

tcnn_eval = tcnn_model.evaluate(X_seq, y_cat)
print("T-CNN Accuracy:", tcnn_eval[1])



