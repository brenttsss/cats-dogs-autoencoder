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

# Extracting features using the encoder
encoded_features = encoder.predict(train_generator)

# Reshaping the encoded features
encoded_features_flat = encoded_features.reshape(encoded_features.shape[0], -1)

# Standardizing the features
scaler = StandardScaler()
encoded_features_flat = scaler.fit_transform(encoded_features_flat)

# Applying PCA for dimensionality reduction (optional)
pca = PCA(n_components=50)
encoded_features_flat = pca.fit_transform(encoded_features_flat)

# Apply Gaussian Mixture Model (GMM) clustering to the latent features
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(encoded_features_flat)
cluster_labels = gmm.predict(encoded_features_flat)

# Visualizing the clusters
plt.scatter(encoded_features_flat[:, 0], encoded_features_flat[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Clustering results')
plt.show()

# Getting one batch of test images
X_test, Y_test = next(test_generator)

# Getting corresponding encoded features for the test set
encoded_test_features = encoder.predict(X_test)
encoded_test_features_flat = encoded_test_features.reshape(encoded_test_features.shape[0], -1)
encoded_test_features_flat = scaler.transform(encoded_test_features_flat)
encoded_test_features_flat = pca.transform(encoded_test_features_flat)

# Predicting clusters for the test set
test_cluster_labels = gmm.predict(encoded_test_features_flat)

# Displaying the first 5 images, their actual labels, and predicted clusters
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(X_test[i].reshape(150, 150, 3))
    plt.title(f"Actual: {int(Y_test[i])} | Cluster: {test_cluster_labels[i]}")
    plt.axis('off')
plt.show()

