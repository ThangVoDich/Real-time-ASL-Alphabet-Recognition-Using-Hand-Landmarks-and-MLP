# Real-time-ASL-Alphabet-Recognition-Using-Hand-Landmarks-and-MLP
This project implements a real-time American Sign Language (ASL) alphabet recognition system using hand landmarks extracted from a webcam feed and a lightweight Multi-Layer Perceptron (MLP) classifier.  Instead of processing raw images, the system relies on geometric hand keypoints, resulting in a fast and efficient pipeline suitable for real-time

Core Idea

Hand images → MediaPipe Hands → 21 hand landmarks

Landmarks → 42-dimensional feature vector (x, y coordinates)

Feature vector → MLP classifier

Output → Predicted ASL letter displayed in real time

 System Pipeline

Landmark Extraction (Offline)

Hand landmarks are extracted from the ASL Alphabet image dataset using MediaPipe.

Each sample is represented by 21 keypoints, resulting in a 42-dimensional vector.

The extracted data is stored in CSV format for training.

Model Training

A Multi-Layer Perceptron (MLP) is trained using TensorFlow/Keras.

The task is formulated as a multi-class classification problem.

The dataset is split into training and validation sets to evaluate performance.

Real-time Inference

Webcam frames are processed to detect hand landmarks.

The trained MLP predicts the corresponding ASL letter in real time.

A prediction buffer is applied to stabilize outputs and reduce noise.

Users can confirm predictions to construct text sequences.

Technologies Used

Python

OpenCV – webcam processing and visualization

MediaPipe Hands – hand landmark detection

NumPy, Pandas – data processing

TensorFlow / Keras – model training

Scikit-learn – label encoding and data splitting
