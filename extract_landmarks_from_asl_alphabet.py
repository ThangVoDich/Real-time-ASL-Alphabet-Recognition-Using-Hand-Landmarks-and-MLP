# extract_landmarks_from_asl_alphabet.py
"""
Extract hand landmarks from real hand images in asl_alphabet_train/ using MediaPipe.
Saves to asl_landmark_data.csv: label, x0, y0, ..., x20, y20
"""
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from tqdm import tqdm

INPUT_DIR = 'asl_alphabet_train'
CSV_FILE = 'asl_landmark_data.csv'
CLASSES = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])

mp_hands = mp.solutions.hands

with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    writer.writerow(header)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for cls in CLASSES:
            input_folder = os.path.join(INPUT_DIR, cls)
            images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in tqdm(images, desc=f'Processing {cls}'):
                img_path = os.path.join(input_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm = [(l.x, l.y) for l in hand_landmarks.landmark]
                    row = [cls] + [coord for point in lm for coord in point]
                    writer.writerow(row)
                else:
                    print(f"No hand detected in: {img_path}")
print('Landmark extraction complete!')
