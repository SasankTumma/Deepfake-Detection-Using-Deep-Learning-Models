import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Check for GPU
if tf.config.list_physical_devices("GPU"):
    print("GPU is available.")
else:
    print("GPU is not available, using CPU.")

# Load parquet files and dataset
folder_path = "./dataset"
parquet_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".parquet")
]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

images, labels = [], []
for _, row in df.iterrows():
    image_data = row["image"]["bytes"]
    label = row["label"]
    image = Image.open(BytesIO(image_data)).resize((227, 227))
    images.append(np.array(image))
    labels.append(label)

images, labels = np.array(images), np.array(labels)
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Split the dataset
X_subset, _, y_subset, _ = train_test_split(
    images, labels, test_size=0.3, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Define the model
input_shape = (227, 227, 3)
inputs = Input(shape=input_shape)
x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu")(inputs)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Conv2D(256, kernel_size=(5, 5), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(384, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=inputs, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
batch_size = 16
epochs = 10
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
)

# Plot training history
plt.figure()
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


model.save("alexnet_model.h5")