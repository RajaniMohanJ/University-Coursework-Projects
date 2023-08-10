# -*- coding: utf-8 -*-
"""CV_CW_SVM_MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YIXB287vfcKIls20r9nzeGoeYGQQwNBt

# Computer Vision Coursework Submission (IN3060/INM460)

**Student name, ID and cohort:** Rajani Mohan Janipalli (210049506) - PG

## Training and testing script of SIFT-SVM, SIFT-MLP, HOG-SVM and HOG-MLP models.

**Google Colab Setup**
"""

from google.colab import drive
drive.mount('/content/drive')

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""**Updating Open CV**"""

!pip install opencv-python==4.5.5.64

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""**Check the version open CV**"""

!pip show opencv-python

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""**Assign path to link the folder containing the colab notebook.**"""

import os


GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Code' 
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print(os.listdir(GOOGLE_DRIVE_PATH))


## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""**Assign path to link the folder containing the train dataset.**"""

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE_DS = 'Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/CW_Dataset' 
GOOGLE_DRIVE_PATH_DS = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE_DS)
print(os.listdir(GOOGLE_DRIVE_PATH_DS))


# Code from above cell was modified in the this cell, as the dataset resides in a different folder.

"""**Copy and unzip the dataset directly in colab server.**"""

# Identify path to zipped dataset
CW_zip_path = os.path.join(GOOGLE_DRIVE_PATH_DS, 'CW_Dataset.zip')

# Copy it to Colab
!cp '{CW_zip_path}' .

# Unzip it
!yes|unzip -q CW_Dataset.zip

# Delete zipped version from Colab (not from Drive)
!rm CW_Dataset.zip


## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""**Import necessary libraries.**"""

# Commented out IPython magic to ensure Python compatibility.
import cv2
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte, io, color
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# %matplotlib inline

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

import pickle
import time

"""Create a function to import images & labels from dataset."""

def import_imagelabels_lists(path):
    """Load images from selected directories"""
    images = []

    file_names = [file for file in sorted(os.listdir(os.path.join(path))) if file.endswith('.jpg')]
    for file in file_names:
        images.append(io.imread(os.path.join(path, file)))
    
    label_set = np.loadtxt(os.path.join('labels', 'list_label_{}.txt'.format(path)), dtype='str')
    label_nums = [] # create an empty list to append label numbers.
    for i in range(len(label_set)): # execute a for loop to extract the exact labels from data and append them to a list.
      label_nums.append(label_set[i][1])
    
    label_names = ['Suprise' if p == '1' else 'Fear' if p == '2' else 'Disgust' if p == '3' else 'Happiness' if p == '4' else 'Sadness' if p == '5' else 'Anger' if p == '6' else 'Neutral' for p in label_nums]

    return images,label_names


## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module
## and modified as per the requirement in this task.
## https://tutorial.eyehunts.com/python/python-elif-in-list-comprehension-conditionals-example-code/

"""Import training set images & labels."""

X_1, y_1 = import_imagelabels_lists('train')

type(X_1) # Check the data type.

len(X_1) # Check the total number of images in train data.

"""*Above output showing the total number of images in train data matches with that menstioned in the coursework specification document.*"""

type(X_1[0])

X_1[0] # Check an element of train features to verify that it is an image array.

plt.imshow(X_1[0]) # plot an element of train features data to see the image.

"""Label explanation: (taken from Readme file of dataset)

1: Surprise

2: Fear

3: Disgust

4: Happiness

5: Sadness

6: Anger

7: Neutral
"""

type(y_1)

len(y_1)

"""*The above output shows that the total number of labels in train target matches with the total number of images in the train features.*"""

y_1[:10]

"""Value counts of labels."""

print(Counter(y_1))

"""*Clearly, the dataset is imbalanced.*

Plot first 10 images with their labels.
"""

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_1[i])
    ax[i].set_title(f'Label: {y_1[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.



# X_train, X_val, y_train, y_val = train_test_split(X, y_12, test_size=0.1, shuffle=True, stratify=y_12)

# ## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

X_train = X_1
y_train = y_1

len(X_train)

len(y_train)

"""Extract SIFT descriptors for training data."""

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Create empty lists for feature descriptors and labels
des_list = []
y_train_list = []

fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)

for i in range(len(X_train)):
    # Identify keypoints and extract descriptors with SIFT
    img = img_as_ubyte(color.rgb2gray(X_train[i]))
    kp, des = sift.detectAndCompute(img, None)

    # Show results for first 4 images
    if i<4:
        img_with_SIFT = cv2.drawKeypoints(img, kp, img)
        ax[i].imshow(img_with_SIFT)
        ax[i].set_axis_off()

    # Append list of descriptors and label to respective lists
    if des is not None:
        des_list.append(des)
        y_train_list.append(y_train[i])

fig.tight_layout()
plt.show()

# Convert to array for easier handling
des_array = np.vstack(des_list)


## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Perform KMeans clustering of SIFT descriptors to obtain codewords or Bag of Visual Words."""

# Number of centroids/codewords: good rule of thumb is 10*num_classes
k = len(np.unique(y_train)) * 10

# Use MiniBatchKMeans for faster computation and lower memory usage
batch_size = des_array.shape[0] // 4
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Save the KMeans model to use it for extracting SIFT descriptors for test data."""

import pickle

siftKMeans = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/siftKMeans.pickle'

with open (siftKMeans, 'wb') as f:
    pickle.dump(kmeans, f)

print(k)

"""Find the optimal number of clusters from elbow diagram."""

k_rng = [10, 30, 50, 70, 90, 100]
sse = []
for knum in k_rng:
  kmelb = MiniBatchKMeans(n_clusters=knum, batch_size=batch_size).fit(des_array)
  sse.append(kmelb.inertia_)

  ## CODING REFERENCE:
  ## https://www.youtube.com/watch?v=EItlUEPCIzM

len(des_array)

plt.plot(k_rng, sse)
plt.xlabel('K')
plt.ylabel('Sum of squared error')

k_rng1 = [10, 50, 100, 150, 200]
sse1 = []
for knum1 in k_rng1:
  kmelb1 = MiniBatchKMeans(n_clusters=knum1, batch_size=batch_size).fit(des_array)
  sse1.append(kmelb1.inertia_)

plt.plot(k_rng1, sse1)
plt.xlabel('K')
plt.ylabel('Sum of squared error')

k_rng2 = [50, 100, 200, 300, 400]
sse2 = []
for knum2 in k_rng2:
  kmelb2 = MiniBatchKMeans(n_clusters=knum2, batch_size=batch_size).fit(des_array)
  sse2.append(kmelb2.inertia_)

plt.plot(k_rng2, sse2)
plt.xlabel('K')
plt.ylabel('Sum of squared error')

"""*All the above search iterations show that 70 is nearly the optimal number of clusters.*"""

# Convert descriptors into histograms of codewords for each image
hist_list = []
idx_list = []

for des in des_list:
    hist = np.zeros(k)

    idx = kmeans.predict(des)
    idx_list.append(idx)
    for j in idx:
        hist[j] = hist[j] + (1 / len(des))
    hist_list.append(hist)

hist_array = np.vstack(hist_list)

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

# fig, ax = plt.subplots(figsize=(8, 3))
# ax.hist(np.array(idx_list, dtype=object), bins=k)
# ax.set_title('Codewords occurrence in training set')
# plt.show()

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""*NOTE: The above cell produces a histograms of codewords generated from feature descriptors. But it has been commented as it always crashes the RAM of colab.*

Create baseline SVM model using SIFT descriptors.
"""

# Create a classifier: a support vector classifier
baseline_siftsvm = svm.SVC(kernel='rbf', class_weight='balanced')

# We learn the digits on the first half of the digits
# baseline_siftsvm.fit(hist_array, y_train_list)

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

from sklearn.model_selection import cross_val_score

"""Perform K-Fold Crossvalidation of the model befor fitting the complete traning data to the model."""

baseline_siftsvm_cvscores = cross_val_score(baseline_siftsvm, hist_array, y_train_list, cv=5)

## CODING REFERENCE:

## https://scikit-learn.org/stable/modules/cross_validation.html

baseline_siftsvm_cvscores

baseline_siftsvm_cvscores.mean()

baseline_siftsvm.fit(hist_array, y_train_list)

baseline_siftsvm.score(hist_array, y_train_list)

baseline_siftsvm.get_params()

"""Perform exhaustive grid search for tuning hyperparameters."""

from sklearn.model_selection import GridSearchCV

siftsvm_improve1 = svm.SVC(class_weight='balanced')

siftsvm_gs1 = GridSearchCV(estimator=siftsvm_improve1,
             param_grid={'C': [0.5, 1, 10, 100], 'kernel': ('linear', 'poly','rbf')})


## CODING REFERENCE:
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

siftsvm_gs1.fit(hist_array, y_train_list)

siftsvm_gs1.best_params_

siftsvm_gs1.best_score_

siftsvm_1 = svm.SVC(C=10, kernel='rbf', class_weight='balanced')

siftsvm_1_cvscores = cross_val_score(siftsvm_1, hist_array, y_train_list, cv=5)

siftsvm_1_cvscores

siftsvm_1_cvscores.mean()

siftsvm_1.fit(hist_array, y_train_list)

siftsvm_1.score(hist_array, y_train_list)

"""After hyperparameter tuning done above, create best SVM model that uses SIFT descriptors."""

best_siftSVM_model = svm.SVC(kernel='rbf', class_weight='balanced')

best_siftSVM_model_cvscores = cross_val_score(best_siftSVM_model, hist_array, y_train_list, cv=5)

best_siftSVM_model_cvscores

best_siftSVM_model_cvscores.mean()

import time

t0 = time.time()
best_siftSVM_model.fit(hist_array, y_train_list)
tt = time.time()
print(tt-t0)

best_siftSVM_model.score(hist_array, y_train_list)

"""Save best SVM model that uses SIFT descriptor."""

import pickle

bestsiftsvmpath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_siftSVM_model.pickle'

with open (bestsiftsvmpath, 'wb') as f:
    pickle.dump(best_siftSVM_model, f)

## CODING REFERENCE
## https://www.youtube.com/watch?v=KfnhNlD8WZI





from sklearn.neural_network import MLPClassifier

"""Create baseline MLP model that uses SIFT descriptors."""

baseline_siftMLP = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=True, random_state=1,
                    learning_rate_init=.1)

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

baseline_siftMLP_cvscores = cross_val_score(baseline_siftMLP, hist_array, y_train_list, cv=5)

baseline_siftMLP_cvscores

baseline_siftMLP_cvscores.mean()

baseline_siftMLP.fit(hist_array, y_train_list)

baseline_siftMLP.score(hist_array, y_train_list)

baseline_siftMLP.get_params()

"""Perform hyperparameter tuning."""

siftMLP_improve1 = MLPClassifier(max_iter=200, alpha=1e-4,
                    solver='sgd', random_state=1,
                    learning_rate_init=.1)

siftMLP_gs1 = GridSearchCV(estimator=siftMLP_improve1,
             param_grid={'hidden_layer_sizes': [(50,), (100,), (50,50), (50,100)]})

siftMLP_gs1.fit(hist_array, y_train_list)

siftMLP_gs1.best_params_

siftMLP_gs1.best_score_

siftMLP_improve2 = MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,
                    solver='sgd', random_state=1,
                    learning_rate_init=.1)

siftMLP_gs2 = GridSearchCV(estimator=siftMLP_improve2,
             param_grid={'max_iter': [200, 300, 400, 500]})

siftMLP_gs2.fit(hist_array, y_train_list)

siftMLP_gs2.best_params_

siftMLP_gs2.best_score_

siftMLP_improve3 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200,alpha=1e-4, 
                                 random_state=1, learning_rate_init=.1)

siftMLP_gs3 = GridSearchCV(estimator=siftMLP_improve3,
             param_grid={'activation': ['identity', 'tanh', 'relu'], 'solver': ['sgd', 'adam']})

siftMLP_gs3.fit(hist_array, y_train_list)

siftMLP_gs3.best_params_

siftMLP_gs3.best_score_

siftMLP_improve4 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200,alpha=1e-4,
                    solver='sgd', random_state=1)

siftMLP_gs4 = GridSearchCV(estimator=siftMLP_improve4,
             param_grid={'learning_rate_init': [0.001, 0.01, 0.1]})

siftMLP_gs4.fit(hist_array, y_train_list)

siftMLP_gs4.best_params_

siftMLP_gs4.best_score_

siftMLP_improve5 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200,alpha=1e-4,
                    solver='sgd', learning_rate_init=0.01, random_state=1)

siftMLP_improve5_cvscores = cross_val_score(siftMLP_improve5, hist_array, y_train_list, cv=5)

siftMLP_improve5_cvscores.mean()

siftMLP_improve5.fit(hist_array, y_train_list)

siftMLP_improve5.score(hist_array, y_train_list)

"""Create best MLP model that uses SIFT descriptors."""

best_siftMLP_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200,alpha=1e-4,
                    solver='sgd', verbose=True, learning_rate_init=0.01, random_state=1)

best_siftMLP_model_cvscores = cross_val_score(best_siftMLP_model, hist_array, y_train_list, cv=5)

best_siftMLP_model_cvscores

best_siftMLP_model_cvscores.mean()

t0 = time.time()
best_siftMLP_model.fit(hist_array, y_train_list)
tt = time.time()
print(tt - t0)

best_siftMLP_model.score(hist_array, y_train_list)

"""Save best MLP model that uses SIFT descriptors."""

bestsiftmlppath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_siftMLP_model.pickle'

with open (bestsiftmlppath, 'wb') as f:
    pickle.dump(best_siftMLP_model, f)







from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

"""Extract HOG descriptors for training data."""

hog_images = []
hog_descriptors = []

X_train_gray = [ color.rgb2gray(i) for i in X_train]

for i in range(len(X_train)):
    # Identify keypoints and extract descriptors with SIFT
    img_hog = img_as_ubyte(np.array(X_train_gray[i]))
    fet_des, hog_img = hog(img_hog, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True, multichannel=False)
    hog_descriptors.append(fet_des)
    hog_images.append(hog_img)


fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)

ax[0].axis('off')
ax[0].imshow(X_train_gray[0])
ax[0].set_title('Original image')

# Rescale histogram for better display
hog_img_rescale = exposure.rescale_intensity(hog_images[0], in_range=(0, 10))

ax[1].axis('off')
ax[1].imshow(hog_img_rescale, cmap='gray')
ax[1].set_title('Histogram of Oriented Gradients')
fig.tight_layout()
plt.show()


## CODING REFERENCE: Code was taken from Lab Tutorial 06 of Computer Vision - IN3060/INM460 module
## and modified as per the requirement in this task.
## https://www.kaggle.com/code/manikg/training-svm-classifier-with-hog-features/notebook

len(hog_descriptors)

"""Create baseline SVM model that uses HOG descriptors."""

baseline_hogSVM = svm.SVC(kernel='rbf', class_weight='balanced')

baseline_hogSVM_cvscores = cross_val_score(baseline_hogSVM, hog_descriptors, y_train, cv=5)

baseline_hogSVM_cvscores

baseline_hogSVM_cvscores.mean()

baseline_hogSVM.fit(hog_descriptors, y_train)

baseline_hogSVM.score(hog_descriptors, y_train)

baseline_hogSVM.get_params()

"""Perform hyperparameter tuning."""

hogSVM_improve1 = svm.SVC(class_weight='balanced')

hogSVM_gs1 = GridSearchCV(estimator=hogSVM_improve1,
             param_grid={'C': [0.5, 1, 10, 100], 'kernel': ('linear', 'poly','rbf')})

hogSVM_gs1.fit(hog_descriptors, y_train)

hogSVM_gs1.best_params_

hogSVM_gs1.best_score_

hogSVM_imporve1 = svm.SVC(C=10, kernel='rbf', class_weight='balanced')

hogSVM_imporve1_cvscores = cross_val_score(hogSVM_imporve1, hog_descriptors, y_train, cv=5)

hogSVM_imporve1_cvscores

hogSVM_imporve1_cvscores.mean()

hogSVM_imporve1.fit(hog_descriptors, y_train)

hogSVM_imporve1.score(hog_descriptors, y_train)

"""Create best SVM model that uses HOG descriptors."""

best_hogSVM_model = svm.SVC(kernel='rbf', class_weight='balanced')

best_hogSVM_model_cvscores = cross_val_score(best_hogSVM_model, hog_descriptors, y_train, cv=5)

best_hogSVM_model_cvscores

best_hogSVM_model_cvscores.mean()

t0 = time.time()
best_hogSVM_model.fit(hog_descriptors, y_train)
tt = time.time()
print(tt - t0)

best_hogSVM_model.score(hog_descriptors, y_train)

"""Save best SVM model that uses HOG descriptors."""

besthogsvmpath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_hogSVM_model.pickle'

with open (besthogsvmpath, 'wb') as f:
    pickle.dump(best_hogSVM_model, f)

"""Create baseline MLP model that uses HOG descriptors."""

baseline_hogMLP = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=True, random_state=1,
                    learning_rate_init=.1)

baseline_hogMLP_cvscores = cross_val_score(baseline_hogMLP, hog_descriptors, y_train, cv=5)

baseline_hogMLP_cvscores

baseline_hogMLP_cvscores.mean()

baseline_hogMLP.fit(hog_descriptors, y_train)

baseline_hogMLP.score(hog_descriptors, y_train)

"""Perform hyperparameter tuning."""

hogMLP_improve1 = MLPClassifier(max_iter=200, alpha=1e-4,
                    solver='sgd', random_state=1,
                    learning_rate_init=.1)

hotMLP_gs1 = GridSearchCV(estimator=hogMLP_improve1,
             param_grid={'hidden_layer_sizes': [(50,), (100,), (50,50), (50,100)]})

hotMLP_gs1.fit(hog_descriptors, y_train)

hotMLP_gs1.best_params_

hotMLP_gs1.best_score_

hogMLP_improve2 = MLPClassifier(hidden_layer_sizes=(100,), alpha=1e-4,
                    solver='sgd', random_state=1,
                    learning_rate_init=.1)

hogMLP_gs2 = GridSearchCV(estimator=hogMLP_improve2,
             param_grid={'max_iter': [200, 300, 400, 500]})

hogMLP_gs2.fit(hog_descriptors, y_train)

hogMLP_gs2.best_params_

hogMLP_gs2.best_score_

hogMLP_improve3 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, 
                                alpha=1e-4,random_state=1,
                    learning_rate_init=.1)

hogMLP_gs3 = GridSearchCV(estimator=hogMLP_improve3,
             param_grid={'activation': ['identity', 'tanh', 'relu'], 'solver': ['sgd', 'adam']})

hogMLP_gs3.fit(hog_descriptors, y_train)

hogMLP_gs3.best_params_

hogMLP_gs3.best_score_

hogMLP_improve4 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200,alpha=1e-4,
                    solver='sgd', random_state=1)

hogMLP_gs4 = GridSearchCV(estimator=hogMLP_improve4,
             param_grid={'learning_rate_init': [0.001, 0.01, 0.1]})

hogMLP_gs4.fit(hog_descriptors, y_train)

hogMLP_gs4.best_params_

hogMLP_gs4.best_score_

hogMLP_improve5 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=600,alpha=1e-4,
                    solver='sgd', learning_rate_init=0.01, random_state=1)

hogMLP_improve5_cvscores = cross_val_score(hogMLP_improve5, hog_descriptors, y_train, cv=5)

hogMLP_improve5_cvscores

hogMLP_improve5_cvscores.mean()

hogMLP_improve5.fit(hog_descriptors, y_train,)

hogMLP_improve5.score(hog_descriptors, y_train,)

"""Create best MLP model that uses HOG descriptors."""

best_hogMLP_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200,alpha=1e-4,
                    solver='sgd', learning_rate_init=0.01, verbose=True, random_state=1)

best_hogMLP_model_cvscores = cross_val_score(best_hogMLP_model, hog_descriptors, y_train, cv=5)

best_hogMLP_model_cvscores

best_hogMLP_model_cvscores.mean()

t0 = time.time()
best_hogMLP_model.fit(hog_descriptors, y_train)
tt = time.time()
print(tt - t0)

best_hogMLP_model.score(hog_descriptors, y_train)

"""Save best MLP model that uses HOG descriptors."""

besthogmlppath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_hogMLP_model.pickle'

with open (besthogmlppath, 'wb') as f:
    pickle.dump(best_hogMLP_model, f)

"""Import images and labels from test data."""

X_2, y_2 = import_imagelabels_lists('test')

type(X_2)

len(X_2)

type(y_2)

len(y_2)

"""*The above shown number of images and labels matches with that mentioned in the coursework specification document.*

Plot first 10 images of test data along with their labels.
"""

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_2[i])
    ax[i].set_title(f'Label: {y_2[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

X_test = X_2
y_test = y_2

testk = len(np.unique(y_test)) * 10

# Use MiniBatchKMeans for faster computation and lower memory usage
batch_size = des_array.shape[0] // 4
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)

"""Extract SIFT descriptors for test data."""

testhist_list = []

for i in range(len(X_test)):
    testimg = img_as_ubyte(color.rgb2gray(X_test[i]))
    testkp, testdes = sift.detectAndCompute(testimg, None)

    if testdes is not None:
        testhist = np.zeros(k)

        testidx = kmeans.predict(testdes)

        for j in testidx:
            testhist[j] = testhist[j] + (1 / len(testdes))

        # hist = scale.transform(hist.reshape(1, -1))
        testhist_list.append(testhist)

    else:
        testhist_list.append(None)

# Remove potential cases of images with no descriptors
testidx_not_empty = [i for i, x in enumerate(testhist_list) if x is not None]
testhist_list = [testhist_list[i] for i in testidx_not_empty]
y_test1 = [y_test[i] for i in testidx_not_empty]
testhist_array = np.vstack(testhist_list)

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

len(y_test1)

"""Load best SVM model that uses SIFT descriptors."""

bestsiftsvmpath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_siftSVM_model.pickle'

with open (bestsiftsvmpath, 'rb') as f:
    best_siftSVM_model_loaded = pickle.load(f)

"""Make predictions from test features."""

y_pred = best_siftSVM_model_loaded.predict(testhist_array).tolist()

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Plot first 10  images of test data along with their actual and predicted labels."""

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test[i])
    ax[i].set_title(f'Label: {y_test1[i]} \n Prediction: {y_pred[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Calculate test score of best SVM model that uses SIFT descriptors."""

siftSVM_test_score = best_siftSVM_model_loaded.score(testhist_array, y_test1)
siftSVM_test_score

"""Generate a classification report of the best SVM model that uses SIFT descriptors."""

print(f"""Classification report for classifier best siftSVM- {best_siftSVM_model_loaded}:
      {metrics.classification_report(y_test1, y_pred)}\n""")

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Plot Confusion matrix for best SVM model that uses SIFT descriptors."""

metrics.ConfusionMatrixDisplay.from_predictions(y_test1, y_pred)
plt.show()

## CODING REFERENCE: Code was taken from Lab Tutorial 07 of Computer Vision - IN3060/INM460 module.

"""Load best MLP model that uses SIFT descriptors and then evaluate its performance over test data."""

bestsiftmlppath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_siftMLP_model.pickle'

with open (bestsiftmlppath, 'rb') as f:
    best_siftMLP_model_loaded = pickle.load(f)

y_pred_siftMLP = best_siftMLP_model_loaded.predict(testhist_array).tolist()

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test[i])
    ax[i].set_title(f'Label: {y_test1[i]} \n Prediction: {y_pred_siftMLP[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

siftMLP_test_score = best_siftMLP_model_loaded.score(testhist_array, y_test1)
siftMLP_test_score

print(f"""Classification report for classifier best siftMPL- {best_siftMLP_model_loaded}:
      {metrics.classification_report(y_test1, y_pred_siftMLP)}\n""")

metrics.ConfusionMatrixDisplay.from_predictions(y_test1, y_pred_siftMLP)
plt.show()

"""Extract HOG descriptors for test data."""

testhog_images = []
testhog_descriptors = []

X_test_gray = [ color.rgb2gray(i) for i in X_test]

for i in range(len(X_test)):
    # Identify keypoints and extract descriptors with SIFT
    testimg_hog = img_as_ubyte(np.array(X_test_gray[i]))
    testfet_des, testhog_img = hog(testimg_hog, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True, multichannel=False)
    testhog_descriptors.append(testfet_des)
    testhog_images.append(testhog_img)


fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)

ax[0].axis('off')
ax[0].imshow(X_test_gray[0])
ax[0].set_title('Original image')

# Rescale histogram for better display
testhog_img_rescale = exposure.rescale_intensity(testhog_images[0], in_range=(0, 10))

ax[1].axis('off')
ax[1].imshow(testhog_img_rescale, cmap='gray')
ax[1].set_title('Histogram of Oriented Gradients')
fig.tight_layout()
plt.show()


## CODING REFERENCE: Code was taken from Lab Tutorial 06 of Computer Vision - IN3060/INM460 module
## and modified as per the requirement in this task.
## https://www.kaggle.com/code/manikg/training-svm-classifier-with-hog-features/notebook

"""Load best SVM model that uses HOG descriptors and then evaluate its performance over test data."""

besthogsvmpath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_hogSVM_model.pickle'

with open (besthogsvmpath, 'rb') as f:
    best_hogSVM_model_loaded = pickle.load(f)

y_pred_hogSVM = best_hogSVM_model_loaded.predict(testhog_descriptors)

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test[i])
    ax[i].set_title(f'Label: {y_test[i]} \n Prediction: {y_pred_hogSVM[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

hogSVM_test_score = best_hogSVM_model_loaded.score(testhog_descriptors, y_test)
hogSVM_test_score

print(f"""Classification report for classifier best hogSVM- {best_hogSVM_model_loaded}:
      {metrics.classification_report(y_test, y_pred_hogSVM)}\n""")

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_hogSVM)
plt.show()

"""Load best MLP model that uses HOG descriptors and then evaluate its performance over test data."""

besthogmlppath = 'drive/My Drive/Colab Notebooks/Computer Vision Coursework/CW_Folder_PG/Models/best_hogMLP_model.pickle'

with open (besthogmlppath, 'rb') as f:
    best_hogMLP_model_loaded = pickle.load(f)

y_pred_hogMLP = best_hogMLP_model_loaded.predict(testhog_descriptors)

fig, axes = plt.subplots(2, 5, figsize=(14, 7), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(10):
    ax[i].imshow(X_test[i])
    ax[i].set_title(f'Label: {y_test[i]} \n Prediction: {y_pred_hogMLP[i]}')
    ax[i].set_axis_off()
fig.tight_layout()
plt.show()

hogMLP_test_score = best_hogMLP_model_loaded.score(testhog_descriptors, y_test)
hogMLP_test_score

print(f"""Classification report for classifier best hogMLP- {best_hogMLP_model_loaded}:
      {metrics.classification_report(y_test, y_pred_hogMLP)}\n""")

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_hogMLP)
plt.show()