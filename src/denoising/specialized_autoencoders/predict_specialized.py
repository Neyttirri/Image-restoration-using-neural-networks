#!/usr/bin/env python

import numpy as np
from PIL import Image 
import PIL 
import joblib
import json
import cv2
import os
import time
import argparse
import sys
from keras.models import load_model

sys.path.append('..\\..\\..\\utils')
from utils import preprocess_input_image
sys.path.append('..\\..\\noise_analysis\\common')
from pca_models import PCA_Noise_Type_Classifier

# command line args
parser = argparse.ArgumentParser(description="specialized denoising eutoencoders",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", type=str, help="path to image to predict")
parser.add_argument("output", type=str, help="path to destination for reconstructed image")
args = vars(parser.parse_args())
src = args['input']
output_dir=args['output']

path_to_models = '..\\..\\..\\checkpoints'
if os.path.exists(os.path.join(path_to_models, 'classifier.joblib')):
    classifier = joblib.load(os.path.join(path_to_models, 'classifier.joblib'))
else:
    print('Error: train noise type classifier first')
    exit(1)
    
models = ['Denoising_Autoencoder_gaussian.h5', 'Denoising_Autoencoder_impulse.h5', 'Denoising_Autoencoder_speckle.h5']

# classify present noise
start_time = time.time()

im = preprocess_input_image(src)
predicted_noise_type, predicted_probabilities = classifier.predict(im)
predicted_noise_type = predicted_noise_type[0].astype(int)
print('predicted prob: ', predicted_probabilities)

print('predicted: ', predicted_noise_type)
if predicted_noise_type == 3:
    if predicted_probabilities[0][3] < 0.85:
        predicted_probabilities = np.delete(predicted_probabilities, 3)
        predicted_noise_type = np.argmax(predicted_probabilities, axis=0)
        print('second highest probability: ', predicted_noise_type)
    else:
        print('No noise detected in image. Exiting...')
        exit(1)
try:
    model = load_model(os.path.join(path_to_models, models[predicted_noise_type]))  
except:
    print(f'no trained model {models[predicted_noise_type]} found. Exiting...')
    exit(1)
im = preprocess_input_image(src, 256)
prediction = model.predict(im)
end_time = time.time()
exec_time = end_time - start_time
print('Execution time: ', exec_time, ' seconds')

# write to file 
prediction = np.uint8(np.round(np.clip(prediction, 0, 1) * 255.))
Image.fromarray(np.squeeze(prediction)).save(output_dir)


