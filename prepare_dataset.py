import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics
from clustimage import Clustimage
path = './dataset_v1/'
training_data=[]
# Load the cascade
counter = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for img in os.listdir(path):
    # read every picture from the dataset
    pic = cv2.imread(os.path.join(path,img))
    # gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    # Detect the faces
    faces = face_cascade.detectMultiScale(pic, 1.4, 10)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #get every cropped face
        cropped_image = pic[y:h + y, x:w + x]
        cropped_image = cv2.resize(cropped_image,(64,64))
        cv2.imwrite("./dataset_croped/{}.jpeg".format(counter), cropped_image)
        counter+=1
        print("storing pic nr {}".format(counter))
        cv2.imwrite("./dataset_croped/{}.jpeg".format(counter),cv2.flip(cropped_image, 1))
        counter+=1
        print("storing pic nr {}".format(counter))

