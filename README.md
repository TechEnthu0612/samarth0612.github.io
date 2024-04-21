# Music Genre Classification
This project aims to classify music genres using machine learning techniques. It utilizes the GTZAN dataset for music genre classification available on Kaggle.

Dataset

The dataset used in this project is the "GTZAN Dataset Music Genre Classification" by andradaolteanu, accessible on Kaggle here. It consists of audio excerpts of 30 seconds each from various music genres including Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, and Rock.

## Google Colab Setup

Open Google Colab and create a new notebook. <br/>
Mount Google Drive to access files and datasets: <br/>
   from google.colab import drive <br/>
   drive.mount('/content/drive') <br/>
Install and import the required libraries: <br/>
!pip install pandas numpy seaborn matplotlib scikit-learn librosa kaggle <br/>

import os <br/>
import pandas as pd <br/>
import numpy as np <br/>
import seaborn as sns <br/>
import matplotlib.pyplot as plt <br/>
import sklearn <br/>
import librosa <br/>
import librosa.display <br/>
from kaggle.api.kaggle_api_extended import KaggleApi <br/>
Set up Kaggle API credentials: <br/>
os.environ["KAGGLE_USERNAME"] = "b20cs048" <br/>
os.environ["KAGGLE_KEY"] = "24a550be5e552e966e4d4b6a2a6bfa51" <br/>
Download the dataset using Kaggle API: <br/>
api = KaggleApi() <br/>
api.authenticate() <br/>
dataset_name = "andradaolteanu/gtzan-dataset-music-genre-classification" <br/>
api.dataset_download_files(dataset_name, path="/content/data", unzip=True) <br/>
Then done the code for all the tasks that has to be done for the project i.e. Music Genre Classification <br/>

Used 6 different classifiers for doing the classification and found the accuracies. <br/>

1. Decision Trees (DecisionTreeClassifier) -> 0.6616
2. K-Nearest Neighbors (KNN Classifier) ->0.8313
3. Naive Bayes (GaussianNB) -> 0.5230
4. Stochastic Gradient Descent (SGDClassifier) -> 0.6591
5. Random Forest (RandomForestClassifier) -> 0.8168
6. Support Vector Machine (SVC) -> 0.7587
   
## Usage
The notebook file contains the complete code for data download, preprocessing, model implementation, evaluation, and visualization.

## Results
The project is done by evaluating several machine learning models including Naive Bayes, Stochastic Gradient Descent, K-Nearest Neighbors, Decision Trees, Random Forest, and Support Vector Machine. <br/>
Model performance is assessed using accuracy scores, confusion matrices, learning curves, and feature importance analysis.
A comparison of model accuracies and class distribution is provided to understand the effectiveness of each model and find that how good project has been done. <br/>

## References
https://www.kaggle.com/code/kraftyeaton/ml-final <br/>
https://www.kaggle.com/code/eashana/music-genre-classification-using-ml <br/>
https://www.kaggle.com/code/aftereffect/musicgenreclassificationfinal <br/>
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br/>
https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/ <br/>
https://www.machinelearningnuggets.com/decision-trees-and-random-forests/ <br/>

Group Members <br/>
1. Rahul Prajapati
2. Rahul Saha
3. Samarth Sala
4. Muskan Gautam
 
