import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.mixture import GaussianMixture
os.environ["OMP_NUM_THREADS"] = "7"

# Checking GPU availability for GPU acceleration
print('Num GPUs available:', len(tf.config.experimental.list_physical_devices('GPU')))

URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# Loading the dataset containing images from a URL and extracting it
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)

# Creating the directories
base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'validation')

model_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Creating the training generator that will be used to train the autoencoder
train_generator = model_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='input',
    subset='training'
)

validation_generator = model_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='input',
    subset='validation'
)

# Creating the generator that will allow for testing with unseen images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    shuffle=True,
    class_mode='binary'
)

# Creating the autoencoder layers
input_img = Input(shape=(150, 150, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((3, 3), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((1, 1), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((1, 1))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='rmsprop', loss='mse')
autoencoder.summary()

# Training the autoencoder
history = autoencoder.fit(
    train_generator,
    steps_per_epoch=16,  # 1600 images = batch_size * steps
    epochs=50,
    validation_data=validation_generator,
    validation_steps=4,  # 400 images = batch_size * steps
    shuffle=True,
    verbose=2)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Creating the encoder model
encoder = Model(input_img, encoded)

# Extract features using the encoder
encoded_features = encoder.predict(test_generator)

# Reshape and scale the features
encoded_features_flat = encoded_features.reshape(encoded_features.shape[0], -1)
scaler = StandardScaler()
encoded_features_flat = scaler.fit_transform(encoded_features_flat)

# Separate features and labels for cats and dogs
cats_features = []
dogs_features = []

for i in range(len(test_generator)):
    batch, labels = next(test_generator)
    encoded_batch = encoder.predict(batch)
    flat_features = encoded_batch.reshape(encoded_batch.shape[0], -1)

    for j in range(flat_features.shape[0]):
        if labels[j] == 0:  # Assuming 0 is for cats
            cats_features.append(flat_features[j])
        else:  # Assuming 1 is for dogs
            dogs_features.append(flat_features[j])

# Convert lists to arrays
cats_features = np.array(cats_features)
dogs_features = np.array(dogs_features)

# Dimensionality reduction for visualization (using PCA)
pca = PCA(n_components=2)
cats_pca = pca.fit_transform(cats_features)
dogs_pca = pca.fit_transform(dogs_features)

# Alternatively, you can use t-SNE for non-linear dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# cats_pca = tsne.fit_transform(cats_features)
# dogs_pca = tsne.fit_transform(dogs_features)

# Plot the latent space for cats
plt.figure(figsize=(8, 6))
plt.scatter(cats_pca[:, 0], cats_pca[:, 1], color='blue', label='Cats')
plt.title('Latent Space for Cats')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()

# Plot the latent space for dogs
plt.figure(figsize=(8, 6))
plt.scatter(dogs_pca[:, 0], dogs_pca[:, 1], color='red', label='Dogs')
plt.title('Latent Space for Dogs')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()

# Combine the data and plot the combined latent space
combined_pca = np.vstack((cats_pca, dogs_pca))
combined_labels = np.array([0] * len(cats_pca) + [1] * len(dogs_pca))  # 0 for cats, 1 for dogs

plt.figure(figsize=(8, 6))
plt.scatter(combined_pca[:len(cats_pca), 0], combined_pca[:len(cats_pca), 1], color='blue', label='Cats')
plt.scatter(combined_pca[len(cats_pca):, 0], combined_pca[len(cats_pca):, 1], color='red', label='Dogs')
plt.title('Combined Latent Space for Cats and Dogs')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()
