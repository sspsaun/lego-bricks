import cv2
import os
import glob
import numpy as np
import tqdm as t
from skimage.feature import hog
import sklearn.neighbors as sn
import sklearn.metrics as sm
import matplotlib.pyplot as plt

featureTr = []
labelTr = []
count = 1

for folderPath in os.listdir('dataset/train'):
    for imgPath in glob.glob('dataset/train/' + folderPath + '/*.png'):
        print(count)
        labelTr.append(folderPath)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
        fd = hog(imgResize, 
                orientations=9, 
                pixels_per_cell=(8, 8), 
                cells_per_block=(3, 3))
        featureTr.append(fd)
        count += 1

# Testing Image Loader and Feature Extraction
for imgPath in glob.glob('dataset/test/*.png'):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)

    fdTest = hog(imgResize, 
                orientations=9, 
                pixels_per_cell=(8, 8), 
                cells_per_block=(3, 3))
    fdTest = np.array(fdTest).reshape(1,-1)

    classifier = sn.KNeighborsClassifier(n_neighbors=1)
    tt = classifier.fit(featureTr, labelTr)
    out = classifier.predict(fdTest)
    head, tail = os.path.split(imgPath)
    print('Test is ' + str(tail) + ' : Answer is ' + str(out) + ' : Accuracy is ??')
    # score = sm.accuracy_score()
    # print('Score is ' + str(score))
