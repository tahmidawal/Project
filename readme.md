Safwan Islam, Tahmid Awal

# Blood Clood Detection using DenseNET
We aim to classify the origins of blood clots in ischemic stroke. We  build a model to differentiate between the two major subtypes of acute ischemic stroke (AIS): cardiac and large artery atherosclerosis. This goal is crucial to better predict stroke etiology, improve post-stroke treatment, and reduce the risk of a second stroke. The dataset is part of the Stroke Thromboembolism Registry of Imaging and Pathology (STRIP) project. By addressing challenges in data format, image file size, and pathology slide quantity, we will contribute to the ongoing efforts to leverage artificial intelligence for stroke management and research.

# How to run and reproduce the results

Accessing Dataset: The dataset should be located on the competition's web page. Look for a "Data" or "Dataset" tab or section. You might need to agree to the competition rules or data use agreement before you can download the dataset.

Downloading Dataset: Once you locate the dataset from here https://www.kaggle.com/competitions/mayo-clinic-strip-ai/data , click the download link or button. The dataset may be in various formats like CSV, JSON, or specific formats like DICOM for medical images, etc. The dataset might be compressed to reduce the download size.

Saving the Dataset: Save the dataset to your local system or cloud environment where you plan to work on the competition. Remember the location where you saved the dataset as you will need to load it into your notebook for analysis and model building.

Loading the Dataset: Inside the notebook, you will need to load the dataset. Depending on the dataset format, this could be done using various Python libraries. For example, if it's a CSV file, you could use the pandas library to load the data as follows:

Environment Setup: First, you need to set up your programming environment. You might require Python and its packages like NumPy, Pandas, Matplotlib, Scikit-learn, Tensorflow or PyTorch. You can install these through pip or conda. Also, make sure Jupyter Notebook is installed.

To install Jupyter Notebook, use pip:

pip install notebook
Download Notebook: Download the challenge notebook to your local system or cloud environment.

Navigate to the folder containing the downloaded notebook in your terminal or command prompt and then start Jupyter Notebook by typing:

This will open a new window in your default web browser that shows the list of notebooks in the folder.

Run Notebook: Click on the challenge notebook to open it. You'll see the code is organized into "cells". To run a cell, click on it and press Shift+Enter. This will execute the current cell and select the one below. Alternatively, you can use the "Run" button in the toolbar.

Run All Cells: If you want to run all cells at once, you can use the "Cell" dropdown menu at the top of the notebook, then select "Run All".

Inspect Results: As you run the cells, outputs (including plots, printed text, etc.) will appear directly below the corresponding cell. If there are any errors, they will also appear here.



# Dataset
The dataset used in this project contains medical images of patients, where each image is labeled as either having the medical condition or not. The dataset is split into a training set and a testing set, with each set containing image files in PNG or TIFF format, as well as a CSV file with the corresponding labels.

# Preprocessing
The first step in the code is to preprocess the dataset by adding new columns to the CSV files that contain the file paths of the images and the corresponding binary labels. The ImageDataGenerator class from the Keras library is then used to generate augmented and preprocessed images from the dataset, which are fed into the model during training.


# Training
The model is trained on the augmented images from the training set using the fit() method of the Keras Model class, with hyperparameters such as the number of epochs, batch size, and learning rate set for the model. Learning rate scheduling and early stopping are implemented using callback functions during training to improve the model's performance and efficiency.

# Evaluation
The performance of the trained model is evaluated on the testing set by computing the accuracy score of the model's predictions using the Scikit-learn library. The predicted probabilities of the model are obtained using the predict() method of the Keras Model class, and are rounded to obtain binary predictions for each image in the testing set.


ChatGPT has been used in this project
