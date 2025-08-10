import cv2
import numpy as np
import os

folder_name = "data"
folder_path = os.path.join(os.getcwd(), folder_name)
# Check if the data folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # Create the folder (and any parents if needed)
    print(f"Folder '{folder_name}' created at: {folder_path}")
else:
    print(f"Folder '{folder_name}' already exists at: {folder_path}")

# initializing camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
face_data=[]
dataset_path = "./data/"

file_name = input ("Enter name of person :")
   
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #saves memory

    if ret == False:
        continue


    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    
    for face in faces[-1:]:  # picking the last face in the sorted array as it would be the largest
        x,y,w,h = face # using w*h we can find the largest face
        cv2.rectangle(frame,(x,y,),(x+w,y+h),(0,255,255),2)

        # Extract - (crop out req face) : Region of Interest
        offset = 10
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # Storing every 10th (largest) face
        skip+=1
        if (skip%10==0):
            face_data.append(face_section)
            print(len(face_data))

        cv2.imshow("Video Frame",frame)
        cv2.imshow("Face",face_section)


    # wait for user input -q, then the loop will stop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Converting out face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Saving this flattened array containing face data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data saved at "+ dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()          
cv2.waitKey(1)