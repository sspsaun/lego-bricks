import cv2
import os
import glob
import numpy as np
import tqdm as t
import sklearn.neighbors as sn
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import skimage.feature as skf

featureTr = []
labelTr = []
count = 1

# settings for LBP
radius = 2
n_points = 8 * radius

# Training Image Loader and Feature Extraction
for folderPath in os.listdir('dataset/train'):
    for imgPath in glob.glob('dataset/train/' + folderPath + '/*.png'):
        print(count)
        labelTr.append(folderPath)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
        lbp = skf.local_binary_pattern(imgResize, n_points, radius)
        lbp = np.array(lbp).reshape(1,-1)
        featureTr.append(lbp[0])
        count += 1

# Testing Image Loader and Feature Extraction
for imgPath in glob.glob('dataset/test/*.png'):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)

    lbp = skf.local_binary_pattern(imgResize, n_points, radius)
    lbp = np.array(lbp).reshape(1,-1)

    classifier = sn.KNeighborsClassifier(n_neighbors=1)
    classifier.fit(featureTr, labelTr)
    out = classifier.predict(lbp)
    head, tail = os.path.split(imgPath)
    print('Test is ' + str(tail) + ' : Answer is ' + str(out) + ' : Accuracy is ??')
