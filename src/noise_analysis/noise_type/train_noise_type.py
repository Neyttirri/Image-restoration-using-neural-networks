#!/usr/bin/env python

import numpy as np
import pandas as pd
from array import *
from PIL import Image, ImageOps
from skimage.util import *
from os.path import isfile, join, exists
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import json
import argparse
import sys

sys.path.append('..\\..\\..\\utils')
from utils import *
sys.path.append('..\\common')
from pca_models import Filter, PCA_Noise_Type_Classifier
# load constants
constants_dictionary = json.load(open("..\\..\\..\\utils\\constants.json","r"))
base_name = constants_dictionary.get('base_name_pca_images')

# optional argument for custom training data
parser = argparse.ArgumentParser(description="train classifier",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--augmented", default=False, type=bool, help="whether the training data should be augmented to increase size")
parser.add_argument("-o", "--overwrite", default=False, type=bool, help="whether pikles should be overwriten, if existing")
args = vars(parser.parse_args())
should_augment_data=args['augmented']
should_overwrite_pkl=args['overwrite']



resize_images_to=64

filter_size_B1=3
filter_size_B2=8

filter_amount_B1=5
filter_amount_B2=10

filter_b1 = Filter(filter_amount_B1, filter_size_B1)
filter_b2 = Filter(filter_amount_B2, filter_size_B2)

# load data
X, y, y_noise_info = get_ds_from_pkl(base_name, resize_images_to=64, overwrite=should_overwrite_pkl)
if should_augment_data:
    X, y_noise_info = get_augmented_images(X,y_noise_info)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_noise_info, 
    test_size=0.3, 
    shuffle=True,
    random_state=42,
)

d4_train = X_train.reshape((X_train.shape[0], 1, resize_images_to, resize_images_to, 1))
d4_test = X_test.reshape((X_test.shape[0],1, resize_images_to, resize_images_to, 1))
print('train dataset shape: ', d4_train.shape)

# train
classifier = PCA_Noise_Type_Classifier(filter_b1, filter_b2)
classifier.fit(d4_train, y_train[:,0])
joblib.dump(classifier, '..\\..\\..\\checkpoints\\classifier.joblib')

# test
print('Test results')
prediction = classifier.predict(d4_test)
classifier.show_confusion_matrix(y_test[:,0], prediction)
