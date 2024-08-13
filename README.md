# Cats and Dogs Autoencoder

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)

## Introduction
This project describes image classification of cats and dogs using an autoencoder for unsupervised learning. Specfically, given a set of images of cats and dogs, the algorithm will first train the model based on regenerating the image using the principal of an autoencoder. Afterwards, with it will implement unsupervised learning to extract the features and then classify images.

## Programs
- **unsupervised_learning.ipynb**: Considers the autoencoder algorithm using a Gaussian mixture model,
- **autoencoder_model_kmeans.py**: Implements the autoencoder algorithm using KMeans,
- **autoencoder_model_gmm.py**: Implements the autoencoder algorithm using a Gaussian mixture model,
- **autoencoder_model_latentspace.py**: Implements the autoencoder algorithm then shows the latent spaces for the two classes.

## Installation
For Tensorflow to work correctly with the version I used (2.10) with GPU support, install the libraries described in requirements.txt.
