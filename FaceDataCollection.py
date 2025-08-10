# Face Data Collection using Haar Cascade

import cv2
import os

# Path to Haar Cascade
cascade_path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

# Directory to store dataset
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Ask for user ID
user_id = input("Enter User ID: ")

cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{dataset_dir}/User.{user_id}.{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 100:  # Stop after 100 images
        break

cap.release()
cv2.destroyAllWindows()

print(f"Data collection complete for User {user_id}")
