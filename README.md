# Real-time ASL Alphabet Recognition Using Hand Landmarks and MLP
 #Overview

This project implements a real-time American Sign Language (ASL) alphabet recognition system using hand landmarks extracted by MediaPipe and a lightweight Multi-Layer Perceptron (MLP) classifier.

Instead of processing raw images, the system relies on geometric hand keypoints, which significantly reduces computational cost and enables real-time inference on CPU.

 ## Key Idea

Hand images → MediaPipe Hands → 21 hand landmarks

Landmarks → 42-dimensional feature vector (x, y coordinates)

Feature vector → MLP classifier

Output → Predicted ASL letter displayed in real time

## System Pipeline

Landmark Extraction

Hand landmarks are extracted from ASL alphabet images using MediaPipe.

Each sample is represented by 21 keypoints (x, y), forming a 42D vector.

The extracted data is saved as a CSV file for training.

## Model Training

A Multi-Layer Perceptron (MLP) is trained using TensorFlow/Keras.

The task is formulated as a multi-class classification problem.

Data is split into training and validation sets.

Real-time Recognition

Webcam input is processed frame by frame.

Hand landmarks are extracted and passed into the trained MLP.

A prediction buffer is applied to stabilize outputs.

Users can confirm predicted letters to build text sequences.

 ## Dataset

The training data is based on the ASL Alphabet Dataset, available on Kaggle:

 ## Kaggle link:
https://www.kaggle.com/datasets/nguynbothng2/asl-alphabet-dataset

Dataset setup

Download the dataset from Kaggle.

Extract it into the project directory as:

data/asl_alphabet_train/

 ## How to Run
### 1️ Install dependencies
pip install -r requirements.txt

### 2️ Extract hand landmarks
python extract_landmarks_from_asl_alphabet.py


This will generate:

asl_landmark_data.csv

### 3️ Train the MLP model
python train_asl_landmark_mlp.py


This will produce:

asl_landmark_mlp_model.h5
asl_landmark_labels.npy

### 4️ Run real-time ASL recognition
python realtime_asl_landmark_mlp.py


Press c to confirm a predicted letter

Press q to quit

## Technologies Used

Python

OpenCV – webcam capture and visualization

MediaPipe Hands – hand landmark detection

NumPy, Pandas – data processing

TensorFlow / Keras – MLP training

Scikit-learn – label encoding and data splitting

