#!/usr/bin/env python

import joblib
import json
import argparse
import os
import sys
import time

sys.path.append('..\\..\\..\\utils')
from utils import preprocess_input_image
sys.path.append('..\\common')
from pca_models import Filter, PCA_Noise_Level_Regressor

# load constants
constants_dictionary = json.load(open("..\\..\\..\\utils\\constants.json","r"))
categorical_noise_types = constants_dictionary.get('categorical_classes_dictionary')

# command line args
parser = argparse.ArgumentParser(description="predict noise level",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", type=str, help="path to image to predict")
args = vars(parser.parse_args())
src = args['input']

# preprocess image
im = preprocess_input_image(src)

path_to_models = '..\\..\\..\\checkpoints'
if os.path.exists(os.path.join(path_to_models, 'classifier.joblib')):
    classifier = joblib.load(os.path.join(path_to_models, 'classifier.joblib'))
else:
    print('Error: train noise type classifier first')
    exit(1)
    
if os.path.exists(os.path.join(path_to_models, 'regressor_3_layered.joblib')):
    regressor = joblib.load(os.path.join(path_to_models, 'regressor_3_layered.joblib'))
else:
    print('Error: no trained model for noise level estimation detected!')
    exit(1)

predicted_noise_type, predicted_noise_probabilities = classifier.predict(im)
noise_name = list(categorical_noise_types.keys())[list(categorical_noise_types.values()).index(predicted_noise_type)]

start_time = time.time()
predicted_noise_level = regressor.predict(im, predicted_noise_type)
end_time = time.time()
elapsed_time = end_time - start_time

print('Execution time testing:', elapsed_time, 'seconds')
print('Predicted noise type: ', noise_name)
print('Predicted noise level: ', predicted_noise_level)
