# Face Recognition using Haar Cascade + LBPH Face Recognizer

import cv2
import numpy as np
import os

# Path to Haar Cascade
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create and train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model(data_path):
    faces = []
    ids = []

    for file_name in os.listdir(data_path):
        if file_name.startswith("User."):
            path = os.path.join(data_path, file_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            user_id = int(file_name.split(".")[1])
            faces.append(img)
            ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    print("Model training complete.")

# Train with dataset
train_model("dataset")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        label = f"User {face_id}" if confidence < 50 else "Unknown"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
