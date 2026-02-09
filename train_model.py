import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(gray_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

faces, ids = getImagesAndLabels("TrainingImage")
recognizer.train(faces, np.array(ids))

os.makedirs("TrainingImageLabel", exist_ok=True)
recognizer.save("TrainingImageLabel/Trainner.yml")

print("Training completed. Trainner.yml created.")
