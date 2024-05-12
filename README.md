# MNIST: Handwritten Digits Classification with Convolutional Neural Networks

![MNIST Digits](https://github.com/symphopkins/MNIST/blob/main/mnist_digits.png)


## Overview
For this project, our objective is to build an optimal convolutional neural network to classify handwritten digits. We will use a built-in dataset, MNIST, provided by TensorFlow. The dataset contains 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

## Files Included

- `MNIST_Digist_Classification.py`: Python script containing the code for loading the dataset, building CNN models, training, and evaluation.
- `MNIST_Digist_Classification.ipynb`: Google Colab Notebook containing the detailed code implementation and explanations.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `LICENSE.txt`: Text file containing the license information for the project.

## Installation
To run this project, ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

## Usage
1. The project starts by loading the MNIST dataset using TensorFlow's Keras API. The data is preprocessed, including normalization of pixel values and reshaping to fit the input requirements of the CNN.
2. Two CNN models are implemented. The architecture of each model is described in detail in the code. Model 1 and Model 2 differ in the arrangement of convolutional and max-pooling layers.
3. Both models are trained on the training data and evaluated on the test data. Training metrics such as loss and accuracy are monitored during the training process. Additionally, test accuracy, classification reports, and confusion matrices are generated to assess the performance of the models.

## License
Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.
