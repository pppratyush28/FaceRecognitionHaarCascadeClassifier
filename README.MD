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
├── haarcascade_frontalface_default.xml # Haar Cascade file (place here)
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

    The file haarcascade_frontalface_default.xml is available from OpenCV's GitHub:
    https://github.com/opencv/opencv/tree/master/data/haarcascades

    Place it in the project root directory.

## **Usage**
1. **Collect Face Data**
    Run the notebook FaceDataCollection.ipynb:
    Captures multiple images of your face.
    Saves them in a folder named dataset/ with labels.

2. **Train & Recognize Faces**
    Run face_recognition.ipynb:
    Loads the dataset.
    Trains a LBPH face recognizer.
    Recognizes faces from webcam feed in real-time.

## **Notes**
    Make sure your webcam is connected and working.
    Lighting conditions affect accuracy; ensure good lighting.
    Haar Cascade is lightweight but less accurate than deep learning–based methods.

## **License**
This project is licensed under the MIT License. You are free to use, modify, and distribute it.