import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categoricalfolder_path = './dataset'
parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)images = []
labels = []

for index, row in df.iterrows():
    image_dict = row['image']
    image_data = image_dict['bytes']
    label = row['label']
    image = Image.open(BytesIO(image_data)).resize((224, 224))
    images.append(np.array(image))
    labels.append(label)

images = np.array(images)
labels = np.array(labels)num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
import os

gc.collect()
K.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate a specific amount of GPU memory
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Limit to 4GB
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)model = Model(inputs=base_model.input, outputs=predictions)for layer in base_model.layers[-4:]:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])batch_size = 8
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs
)plt.figure()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()model.save('vgg16_model.h5')