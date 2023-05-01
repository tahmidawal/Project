# Medical Condition Detection with MobileNetV2
This code implements a machine learning model for detecting a medical condition in medical images, using the MobileNetV2 architecture and image data augmentation techniques.

# Dataset
The dataset used in this project contains medical images of patients, where each image is labeled as either having the medical condition or not. The dataset is split into a training set and a testing set, with each set containing image files in PNG or TIFF format, as well as a CSV file with the corresponding labels.

# Preprocessing
The first step in the code is to preprocess the dataset by adding new columns to the CSV files that contain the file paths of the images and the corresponding binary labels. The ImageDataGenerator class from the Keras library is then used to generate augmented and preprocessed images from the dataset, which are fed into the model during training.

# Model Architecture
The MobileNetV2 architecture is used as the backbone of the classification model, with the pre-trained weights on the ImageNet dataset used as initial weights for the model. The top layers of the model are rebuilt to include a global average pooling layer, batch normalization layers, and dense layers with ReLU activation functions and a sigmoid activation function for binary classification.

# Training
The model is trained on the augmented images from the training set using the fit() method of the Keras Model class, with hyperparameters such as the number of epochs, batch size, and learning rate set for the model. Learning rate scheduling and early stopping are implemented using callback functions during training to improve the model's performance and efficiency.

# Evaluation
The performance of the trained model is evaluated on the testing set by computing the accuracy score of the model's predictions using the Scikit-learn library. The predicted probabilities of the model are obtained using the predict() method of the Keras Model class, and are rounded to obtain binary predictions for each image in the testing set.

Overall, this code provides a complete pipeline for training and evaluating a MobileNetV2 model for detecting a medical condition in medical images, using a dataset with image files and corresponding labels. The code can be easily adapted to other image classification tasks with similar datasets and can be improved by adjusting the model hyperparameters or using more advanced model architectures.
