import numpy as np
import pandas as pd
from array import *
from PIL import Image, ImageOps
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from numpy import linalg as lg
from skimage.util import *
from skimage.metrics import *
from skimage.filters import laplace, sobel, roberts
from skimage import color, data, measure
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction import image
import tensorflow as tf
import random
import cv2
import math
import json
import os
from os.path import isfile, join, exists
import re
import joblib
import glob
import time

# from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, ReLU
constants_dictionary = json.load(open(os.path.join(os.path.dirname(__file__),"constants.json"),"r"))

pkls_dir = '..\\..\\..\\data'


####################### Image quality metrics var
def get_laplacian_var(img_arr_dataset):
    # assuming dataset is with shape (num_images, w, h, channels)
    var = 0
    for image in range(img_arr_dataset.shape[0]):
        img_arr = img_arr_dataset[image]
        lapl=laplace(img_arr)
        var += lapl.var()
    return var / img_arr_dataset.shape[0]

def get_laplacian_var_image(img_arr):
    return laplace(img_arr).var()

def get_average_threshold(sharp_ds, blurry_ds):
    mean_blur_sharp_ds = get_laplacian_var(sharp_ds)
    mean_blur_blurry = get_laplacian_var(blurry_ds)
    return (mean_blur_ground_truth_grayscale + mean_blur_denoised_grayscale) / (sharp_ds.shape[0] + blurry_ds.shape[0])

def mse(data_1, data_2):
    return np.square(np.subtract(data_1, data_2)).mean()

# returns two arrays
# first array with mean blur variances for noisy and denoised images
# second array with mean psnr between noisy and ground truth and between reconstructed and ground truth
def calculate_psnr_ds(ds1, ds2):
    assert ds1.shape == ds2.shape
    total_psnr = 0 
    for i in range(ds1.shape[0]):
        total_psnr += calculate_psnr_image(ds1[i], ds2[1])
    return total_psnr / ds1.shape[0]

def calculate_psnr_image(img1, img2, max_value=1):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        mse = 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def evaluate_psnr_blur(x, y, model, indicies = []):
    size = x.shape[0]
    if len(indicies) == 0:
        indicies = np.array([size / 3, size / 2, size -1  ]).astype(int)
    with tf.device('/CPU:0'):
        visualize_data(x[indicies], 1,3)
        visualize_data(y[indicies], 1,3)
        
        start_time = time.time
        prediction = model.predict(x, batch_size=1)
        end_time = time.time()
        elapsed_time = end_time - start_time

        visualize_data(prediction[indicies], 1,3)

        #################################   blur ##############################
        blur_noisy = np.array([])
        blur_ground_truth = np.array([])
        blur_denoised = np.array([])

        for i in range(x.shape[0]):
            blur_noisy = np.append(blur_noisy, get_laplacian_var_image(x[i]))
            blur_ground_truth = np.append(blur_ground_truth, get_laplacian_var_image(y[i]))
            blur_denoised = np.append(blur_denoised, get_laplacian_var_image(prediction[i]))


        print('noisy images blur average: ', blur_noisy.mean())
        print('ground truth images blur: ', blur_ground_truth.mean())
        print('reconstructd images blur: ',blur_denoised.mean())


        #################################   psnr  ##############################
        psnr_noisy_clean = np.array([])
        psnr_reconstructed_clean = np.array([])

        for i in range(x.shape[0]):
            image_gr = y[i]
            psnr_noisy_clean = np.append(psnr_noisy_clean, calculate_psnr_image(x[i], image_gr))
            psnr_reconstructed_clean = np.append(psnr_reconstructed_clean, calculate_psnr_image(prediction[i], image_gr))


        print('PSNR noisy - ground truth: ', psnr_noisy_clean.mean())
        print('PSNR reconstructed - ground truth: ', psnr_reconstructed_clean.mean())
        
        print('\nExecution time for predicting ', x.shape[0], ' images: ', elapsed_time, 'seconds')
        
        return [blur_noisy.mean(),blur_denoised.mean()], [psnr_noisy_clean.mean(), psnr_reconstructed_clean.mean()]
   
    
def evaluate_image(noise, variance, model):
    x_test, y_test, y_ground_truth = get_single_noisy_image(noise, variance)
    start_time = time.time
    prediction = model.predict(np.expand_dims((x_test), axis=0))
    end_time = time.time()
    elapsed_time = end_time - start_time

    psnr_noisy = calculate_psnr_image(x_test, y_ground_truth)
    psnr_prediction = calculate_psnr_image(prediction, y_ground_truth)
    blur_noisy = laplace(x_test).var()
    blur_ground_truth = laplace(y_ground_truth).var()
    blur_prediction = laplace(prediction).var()

    print(f'{noise} with {variance} variance')
    print('noisy image blur: ', blur_noisy)
    print('ground truth image blur: ', blur_ground_truth)
    print('reconstructed image blur: ',blur_prediction)

    print('PSNR noisy - ground truth: ', psnr_noisy)
    print('PSNR reconstructed - ground truth: ', psnr_prediction)
    
    print('\nExecution time for predicting single image:', elapsed_time, 'seconds')
    
    
def get_single_noisy_image(noise, variance, resize_images_to=256):
    dictionary = constants_dictionary.get('noise_levels_dictionary') 
    categorical_classes_dictionary = constants_dictionary.get('categorical_classes_dictionary')
    im_path = os.path.join(pkls_dir, 'Lena-size-256x256.png')
    im=Image.open(im_path)
    im= im.resize((resize_images_to, resize_images_to))
    im = ImageOps.grayscale(im)
    im_arr = np.asarray(im)
    ground_truth = im_arr / 255
    values = dictionary.get(variance)
#     noise_var = random.uniform(values.get(noise)['min'], values.get(noise)['max'])
    noise_var = values.get(noise)['min']
    class_label = categorical_classes_dictionary.get(noise)
    if(class_label == 1):
        im_arr = random_noise(im_arr, mode=noise, amount=noise_var, salt_vs_pepper=0.5)
    elif class_label == 3:
        im_arr = im_arr / 255
    else:
        im_arr = random_noise(im_arr, mode=noise, var=noise_var)
    y = [class_label, noise_var]
    noisy = im_arr
    
    return np.expand_dims(np.array(noisy), axis=-1), np.expand_dims(np.array(y), axis=-1), np.expand_dims(np.array(ground_truth), axis=-1)



def load_single_variance_to_pkl(base_name, variance, resize_images_to=256, mode='grayscale', noise_type='all', overwrite=False):
    noise_strenghts =  constants_dictionary.get('VALID_NOISE_STRENGHTS')  
    noise_types =  constants_dictionary.get('VALID_NOISE_TYPES')
    
    if variance not in  noise_strenghts:
        raise ValueError("variance must be one of %r." % noise_strenghts)
    if noise_type not in noise_types:
        raise ValueError("variance must be one of %r." % noise_types)
    src = os.path.join(os.path.dirname(__file__), constants_dictionary.get('path_images'))
    
    categorical_classes_dictionary = constants_dictionary.get('categorical_classes_dictionary')
    noises = constants_dictionary.get('noise_dictionary')
    values = noises.get(variance)
    data = dict()
    data['label'] = []
    data['label_ground_truth'] = []
    data['data'] = [] 
    pklname = f"{base_name + mode}_{variance}_{noise_type}_{resize_images_to}x{resize_images_to}px.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)
        for sub_subdir in os.listdir(current_path):
            noise = sub_subdir if noise_type == 'all' else noise_type
            current_path = os.path.join(current_path, sub_subdir)
            print(current_path)
            print(sub_subdir)
            for file in os.listdir(current_path):
                im=Image.open(os.path.join(current_path, file))
                im= im.resize((resize_images_to, resize_images_to))
                if mode == 'grayscale': 
                    im = ImageOps.grayscale(im)
                im_arr = np.asarray(im)
                if(im_arr.shape == (resize_images_to,resize_images_to)):
                    data['label_ground_truth'].append(im_arr / 255)
                    noise_var = random.uniform(values.get(noise)['min'], values.get(noise)['max'])
                    class_label = categorical_classes_dictionary.get(noise)
                    if(class_label == 1):
                        im_arr = random_noise(im_arr, mode=noise, amount=noise_var, salt_vs_pepper=0.5)
                    elif class_label == 3:
                        im_arr = im_arr / 255
                    else:
                        im_arr = random_noise(im_arr, mode=noise, var=noise_var)
                    y = [class_label, noise_var]
                    data['data'].append(im_arr)
                    data['label'].append(y)
 
    print('dumping: ', os.path.join(pkls_dir, pklname))
    joblib.dump(data, os.path.join(pkls_dir, pklname))
    

    
def load_mixed_ds_to_pkl(base_name, mode='grayscale',resize_to=256, noise_types='all'):
    for var in constants_dictionary.get('VALID_NOISE_STRENGHTS'):
        load_single_variance_to_pkl(base_name, var, resize_images_to=resize_to, mode=mode, noise_type=noise_types)
        
def get_ds_from_pkl(base_name, variance=['mixed'], mode='grayscale', resize_images_to=256, noise_types='all', overwrite=False):
    X = np.array([])
    label_ground_truth = np.array([])
    label_noise = np.array([])
    variances = constants_dictionary.get('VALID_NOISE_STRENGHTS') if variance==['mixed'] else variance
    for var in variances:
        pklname = f"{base_name + mode}_{var}_{noise_types}_{resize_images_to}x{resize_images_to}px.pkl"
        if not exists(os.path.join(pkls_dir, pklname)) or overwrite:
            load_single_variance_to_pkl(base_name, var, resize_images_to, noise_type=noise_types)
        data = joblib.load(os.path.join(pkls_dir, pklname))
        x = np.array(data['data'])
        y = np.array(data['label_ground_truth'])
        y_noise = np.array(data['label'])
        if X.size == 0:
            X = x
            label_ground_truth = y
            label_noise = y_noise
        else:
            X = np.concatenate((X, x), axis = 0)
            label_ground_truth = np.concatenate((label_ground_truth, y), axis = 0)
            label_noise = np.concatenate((label_noise, y_noise), axis = 0)
            
    return X, label_ground_truth, label_noise


def get_augmented_images(images, im_labels):
    augmented = []
    aug_labels = []
    for index in range(images.shape[0]):
        curr_image = images[index]
        curr_label = im_labels[index]
        flipped_lr = np.fliplr(curr_image)
        flipped_up = np.flipud(curr_image)
        rotated = rotate(curr_image, 45, resize=False)
        augmented.extend([curr_image, flipped_lr, flipped_up, rotated])
        aug_labels.extend([curr_label, curr_label, curr_label, curr_label])
    return np.array(augmented), np.array(aug_labels)

def visualize_data(data, row, column, labels=[], mode='grayscale'):
    data = data.reshape(data.shape[0], data.shape[1],data.shape[1]) if mode == 'grayscale' else data.reshape(data.shape[0], data.shape[2],data.shape[2], data.shape[3])
    count = 0
    fig, axes = plt.subplots(row, column, figsize = (32,32))
    label_dictionary = {
        0: 'gaussian',
        1:'impulse',
        2:'speckle',
        3:'clean'
    }
    
    for i in range(row):
        for j in range(column):
            if row != 1:
                axes[i,j].imshow(data[count], cmap = 'gray')
            else:
                axes[j].imshow(data[count], cmap = 'gray')
            if len(labels) > 0:
                label = labels[count]
                axes[i,j].set_title(label_dictionary.get(label[0]) + ' ' + str(label[1]))
            count+=1
        

def crop_center(img,cropx,cropy):
    y,x,n = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def preprocess_input_image(path_to_image, resize_images_to = 64):
    im = Image.open(path_to_image)
    im = im.resize((resize_images_to, resize_images_to))
    im = ImageOps.grayscale(im)
    im_arr = np.asarray(im)

    if im_arr.max() > 1:
        im_arr = im_arr / 255

    return im_arr.reshape((1, resize_images_to, resize_images_to, 1))
