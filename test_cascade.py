import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("Loaded:", not faceCascade.empty())
