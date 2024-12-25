import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pltfolder_path = './dataset'
parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)images = []
for index, row in df.iterrows():
    image_dict = row['image']
    image_data = image_dict['bytes']
    image = Image.open(BytesIO(image_data)).resize((64, 64))
    images.append(np.array(image))

images = np.array(images)
images = (images - 127.5) / 127.5

X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)def build_generator():
    model = Sequential()
    model.add(Dense(8 * 8 * 256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(64, 64, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return modeldiscriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()

z = tf.keras.layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
real_or_fake = discriminator(img)
gan = Model(z, real_or_fake)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
d_losses = []
g_losses = []

batch_size = 32
epochs = 1000
half_batch = batch_size // 2

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_imgs = X_train[idx]
    noise = np.random.normal(0, 1, (half_batch, 100))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

    d_losses.append(d_loss[0])
    g_losses.append(g_loss)

    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
generator = build_generator()z = tf.keras.layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
real_or_fake = discriminator(img)
gan = Model(z, real_or_fake)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')plt.figure()
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('GAN Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

generator.save('gan_generator.h5')
discriminator.save('gan_discriminator.h5')