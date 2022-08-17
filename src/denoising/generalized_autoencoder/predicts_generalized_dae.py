#!/usr/bin/env python

import numpy as np
from PIL import Image 
import PIL 
import joblib
import json
import cv2
import os
import sys
import time
import argparse
from keras.models import load_model

sys.path.append('..\\..\\..\\utils')
from utils import preprocess_input_image


# command line args
parser = argparse.ArgumentParser(description="general denoising autoencoder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", type=str, help="path to image to predict")
parser.add_argument("output", type=str, help="path to destination for reconstructed image")
args = vars(parser.parse_args())
src = args['input']
output_dir=args['output']


start_time = time.time()

# load model
path_to_models = '..\\..\\..\\checkpoints'
if os.path.exists(os.path.join(path_to_models, 'Denoising_Autoencoder_grayscale.h5')):
    dae_grayscale = load_model(os.path.join(path_to_models, 'Denoising_Autoencoder_grayscale.h5'))
else:
    print('Error: no trained model detected!')
    exit(1)
    
# preprocess image
im = preprocess_input_image(src, 256)

# predict
prediction = dae_grayscale.predict(im)
end_time = time.time()
exec_time = end_time - start_time

print('Execution time: ', exec_time, ' seconds')
print('blur reconstructed image: ', get_laplacian_var_image(prediction))
print('blur clean image: ', get_laplacian_var_image(im))

# write to file 
prediction = np.uint8(np.round(np.clip(prediction, 0, 1) * 255.))
Image.fromarray(np.squeeze(prediction)).save(output_dir)



