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
from pca_models import Filter, PCA_Noise_Type_Classifier
# load constants
constants_dictionary = json.load(open("..\\..\\..\\utils\\constants.json","r"))
categorical_noise_types = constants_dictionary.get('categorical_classes_dictionary')

# command line args
parser = argparse.ArgumentParser(description="predict noise type",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", type=str, help="path to image to predict")
args = vars(parser.parse_args())
src = args['input']
print(src)
# preprocess image
im = preprocess_input_image(src)

path_to_model = os.path.join('..\\..\\..\\checkpoints', 'classifier.joblib')
if os.path.exists(path_to_model):
    classifier = joblib.load(path_to_model)
else:
    print('Error: model not detected! Train noise type classifier first')
    exit(1)

start_time = time.time()
prediction, prediction_proba = classifier.predict(im)
end_time = time.time()
elapsed_time = end_time - start_time
print('Execution time testing:', elapsed_time, 'seconds')

print('predicted: ', prediction, ': ', list(categorical_noise_types.keys())[list(categorical_noise_types.values()).index(prediction)])

