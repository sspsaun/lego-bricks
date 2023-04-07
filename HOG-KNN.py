import cv2
import os
import glob
import numpy as np
from skimage.feature import hog
import sklearn.neighbors as sn

featureTr = []
labelTr = []
count = 1

for folderPath in os.listdir('dataset/train'):
    for imgPath in glob.glob('dataset/train/' + folderPath + '/*.png'):
        print(count)
        labelTr.append(folderPath)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hogv = hog(img, 
                orientations=9, 
                pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2))
        featureTr.append(hogv)
        count += 1

# Testing Image Loader and Feature Extraction
for imgPath in glob.glob('dataset/test/*.png'):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hogvt = hog(img, 
                orientations=9, 
                pixels_per_cell=(8, 8), 
                cells_per_block=(3, 3))
    hogvt = np.array(hogvt).reshape(1,-1)
    knnClassifier = sn.KNeighborsClassifier(n_neighbors=1)
    knnClassifier.fit(featureTr, labelTr)
    out = knnClassifier.predict(hogvt)
    head, tail = os.path.split(imgPath)
    print('Test is ' + str(tail) + ' : Answer is ' + str(out))
