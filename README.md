

## MNIST Digit Classifier
This repository contains a Jupyter Notebook implementation of a machine learning model to classify handwritten digits from the MNIST dataset. The MNIST dataset is a widely used benchmark dataset in machine learning, consisting of 28x28 grayscale images of digits (0-9) and their corresponding labels.

### Features

#### Data Preprocessing:
- Normalization of pixel values to the range [0, 1].
- One-hot encoding of target labels for multi-class classification.


#### Model Architecture:
- A neural network model implemented using TensorFlow/Keras.
- Configurable layers and activation functions.


#### Training and Evaluation:
- Model training on the MNIST training dataset.
- Evaluation of accuracy and loss on the test dataset.


#### Visualization:

- Display of sample predictions.
- Visualization of training and validation accuracy/loss curves.


#### Requirements
To run the notebook, you need the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow/Keras
- NumPy
- Matplotlib

#### Install the required libraries using the following command:
- pip install tensorflow numpy matplotlib

#### Usage
#### Clone the repository:

#### Open the Jupyter Notebook:
- MNIST_classification.ipynb

#### Run the cells in the notebook to:
- Preprocess the data.
- Train the model.
- Evaluate the model.
- Visualize the results.

#### Dataset
The MNIST dataset is automatically downloaded using TensorFlow/Keras's built-in dataset utilities. It consists of:

- Training Data: 60,000 images of size (28, 28).
- Test Data: 10,000 images of size (28, 28).


#### Results
The model achieves an accuracy of approximately 99.12% on the test dataset.
Below is an example of predictions made by the model:

<img width="712" alt="image" src="https://github.com/user-attachments/assets/9cd9b084-fc46-4e8e-a136-95c5abc63012" />



#### Contributing
Contributions are welcome! If you have suggestions for improving the model or adding new features, feel free to open an issue or submit a pull request.

