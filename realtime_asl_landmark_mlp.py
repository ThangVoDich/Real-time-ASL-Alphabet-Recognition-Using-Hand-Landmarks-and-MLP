# realtime_asl_landmark_mlp.py
"""
Real-time ASL letter recognition using MediaPipe hand landmarks and trained MLP (with all classes).
"""
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = 'asl_landmark_mlp_model.h5'
LABELS_PATH = 'asl_landmark_labels.npy'

# Load model and labels
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

mp_hands = mp.solutions.hands

def extract_landmark_vector(hand_landmarks):
    return np.array([coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)], dtype='float32')

def main():
    cap = cv2.VideoCapture(0)
    import time
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        buffer = []
        buffer_size = 5
        output_text = ""
        last_letter = ""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            display = frame.copy()
            letter = ""
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm_vec = extract_landmark_vector(hand_landmarks)
                pred = model.predict(lm_vec.reshape(1, -1))
                letter = labels[np.argmax(pred)]
                buffer.append(letter)
                if len(buffer) > buffer_size:
                    buffer.pop(0)
                # Only confirm letter if stable and user presses 'c'
                if buffer.count(buffer[0]) == buffer_size:
                    cv2.putText(display, f"Press 'c' to confirm: {buffer[0]}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), 2)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        if buffer[0] == 'space':
                            output_text += ' '
                            last_letter = 'space'
                        elif buffer[0] == 'del':
                            output_text = output_text[:-1]
                            last_letter = 'del'
                        elif buffer[0] != 'nothing':
                            output_text += buffer[0]
                            last_letter = buffer[0]
                    elif key == ord('q'):
                        break
            else:
                buffer = []
            cv2.putText(display, f'Prediction: {letter}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(display, f'Text: {output_text}', (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('Webcam', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
