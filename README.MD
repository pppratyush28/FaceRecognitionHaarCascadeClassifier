# Face Recognition with OpenCV (Haar Cascade)

This project demonstrates **face data collection** and **face recognition** using OpenCV's Haar Cascade Classifier.  
It consists of two scripts/notebooks:
1. **Face Data Collection** – Captures face images from your webcam and stores them in a dataset.
2. **Face Recognition** – Trains a model using collected images and recognizes faces in real-time.

---

## **Project Structure**
```
├── FaceDataCollection.ipynb
├── face_recognition.ipynb
├── haarcascade_frontalface_default.xml # Haar Cascade file 
├── requirements.txt
└── README.md
```

---

## **Installation**
1. **Clone this repository**

   git clone https://github.com/your-username/face-recognition-haar.git
   cd face-recognition-haar

2. **Install dependencies**

    pip install -r requirements.txt

3. **Download Haar Cascade**

    (To be done if not already cloned from the project)
    The file haarcascade_frontalface_default.xml is available from OpenCV's GitHub:
    https://github.com/opencv/opencv/tree/master/data/haarcascades

    Place it in the project root directory.

## **Usage**
1. **Collect Face Data**
    Run the notebook FaceDataCollection.ipynb:
    It captures multiple images of your face.
    Saves them in a folder named data.

2. **Train & Recognize Faces**
    Run face_recognition.ipynb:
    It loads the dataset.
    Recognizes faces from webcam feed in real-time using KNN Classifier.

## **Notes**
    Make sure your webcam is connected and working.
    Remember to press 'q' on the keyboard to stop the face data collection program.
    Remember to press 'q' on the keyboard when you want to stop the face recognition program.
    Lighting conditions affect accuracy; ensure good lighting.
    Haar Cascade is lightweight but less accurate than deep learning–based methods.

## **License**
This project is licensed under the MIT License. You are free to use, modify, and distribute it.