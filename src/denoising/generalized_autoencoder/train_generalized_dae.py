#!/usr/bin/env python

import numpy as np
import pandas as pd
from array import *
from os.path import isfile, join, exists
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import activations, callbacks
import time
import json
import argparse
import sys 

sys.path.append('..\\..\\..\\utils')
from utils import *
from autoencoder_utils import * 

# load constants
constants_dictionary = json.load(open("..\\..\\..\\utils\\constants.json","r"))
base_name = constants_dictionary.get('base_name_dae_images')
path_to_models = '..\\..\\..\\checkpoints'

# optional argument for custom training data
parser = argparse.ArgumentParser(description="train generalized autoencoder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--augmented", default=False, type=bool, help="whether the training data should be augmented to increase size")
parser.add_argument("-o", "--overwrite", default=False, type=bool, help="whether pikles should be overwriten, if existing")
args = vars(parser.parse_args())
should_augment_data=args['augmented']
should_overwrite_pkl=args['overwrite']



# load data
X, y, y_noise_info = get_ds_from_pkl(base_name,resize_images_to=256, overwrite=should_overwrite_pkl)
if should_augment_data:
    X, y = get_augmented_images(X,y)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    shuffle=True,
    random_state=42,
)

d4_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
d4_test=X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# create model
model = create_model('Denoising_Autoencoder_grayscale')
model.compile(optimizer="adam", loss="mse")
path_to_models = '..\\..\\..\\checkpoints'


# train
print('training model...')
start_time = time.time()
checkpoint = ModelCheckpoint(os.path.join(path_to_models, 'Denoising_Autoencoder_grayscale.h5'), save_best_only=True, save_weights_only=False, verbose=1)
history = model.fit(d4_train, y_train, batch_size=4, epochs=70, callbacks=checkpoint, validation_split=0.25, verbose=2)
end_time = time.time()
elapsed_time = end_time - start_time
print('Execution time training:', elapsed_time, 'seconds')

print('Test results')
prediction = model.predict(d4_test)
print('Average PSNR: ', calculate_psnr_ds(np.squeeze(prediction), y_test))
print('Average blur: ', get_laplacian_var(prediction))
print('Average blur clean images: ', get_laplacian_var(y_test))


