import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from timm import create_model
import numpy as np

# Emotion labels — you can change based on the model you use
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load pre-trained ViT model from timm (adjust based on your setup)
# This example assumes you have fine-tuned a ViT model for emotion detection
# Otherwise, you can use a ViT and fine-tune it on emotion datasets like FER2013
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(emotion_labels))
# Replace above with your fine-tuned model if available
# model.load_state_dict(torch.load('path_to_your_emotion_model.pth'))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = to_pil_image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            label = emotion_labels[pred]

        # Draw result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection - ViT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
