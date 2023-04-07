import cv2
import os
import glob
import numpy as np
import sklearn.neighbors as sn
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import skimage.feature as skf

featureTr = []
labelTr = []
count = 1

paraQuantize = 64 
paraAngle = [0, 45, 90, 135]
paraDistance = [1, 2, 3]

# Training Image Loader and Feature Extraction
for folderPath in os.listdir('dataset/train'):
    for imgPath in glob.glob('dataset/train/' + folderPath + '/*.png'):
        print(count)
        labelTr.append(folderPath)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
        img = (img/(256/paraQuantize)).astype(int); # Image Quantization
        glcm = skf.graycomatrix(img, distances=paraDistance, angles=paraAngle, 
        levels=paraQuantize, symmetric=True, normed=True)
        featureCon = skf.graycoprops(glcm, 'contrast')[0]
        featureEne = skf.graycoprops(glcm, 'energy')[0]
        featureHom = skf.graycoprops(glcm, 'homogeneity')[0]
        featureCor = skf.graycoprops(glcm, 'correlation')[0]
        featureTmp = np.hstack((featureCon, featureEne, featureHom, featureCor))
        featureTr.append(featureTmp)
        count += 1
featureTr = np.array(featureTr)
print(featureTr)
print(featureTr.shape)

# Testing Image Loader and Feature Extraction
for imgPath in glob.glob('dataset/test/*.png'):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgResize = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)

    img = (img/(256/paraQuantize)).astype(int)
    glcm = skf.graycomatrix(img, distances=paraDistance, angles=paraAngle, levels=paraQuantize, 
    symmetric=True, normed=True)
    featureCon = skf.graycoprops(glcm, 'contrast')[0]
    featureEne = skf.graycoprops(glcm, 'energy')[0]
    featureHom = skf.graycoprops(glcm, 'homogeneity')[0]
    featureCor = skf.graycoprops(glcm, 'correlation')[0]
    featureTs = [np.hstack((featureCon, featureEne, featureHom, featureCor))]

    classifier = sn.KNeighborsClassifier(n_neighbors=1)
    classifier.fit(featureTr, labelTr)
    out = classifier.predict(featureTs)
    head, tail = os.path.split(imgPath)
    print('Test is ' + str(tail) + ' : Answer is ' + str(out) + ' : Accuracy is ??')
    