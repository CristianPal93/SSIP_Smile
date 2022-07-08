#importing the libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
path = './dataset'

training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic,(64,64))
    training_data.append([pic])
