from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from array import *
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
from skimage.util import *
from skimage.metrics import *
from numpy import linalg as lg
import os
from os.path import isfile, join, exists
import re

import sklearn
import joblib
from skimage.io import imread
from skimage.transform import resize, rotate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from sklearn.neural_network import MLPClassifier, MLPRegressor
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Layer, Activation
from tensorflow.keras import activations, callbacks
import random
from sklearn import metrics
import time
import json
import sys

sys.path.append('..\\..\\..\\utils')
from utils import *

pkls_dir = '..\\common'


################################### Functions ###################################
def conv2d(inputs, filters):
    return tf.nn.conv2d(inputs, filters , strides=[1, 1, 1, 1] , padding='VALID' ) 
 
def maxpool(inputs):
    if len(inputs.shape) > 4:
        inputs = np.squeeze(inputs)
    return tf.nn.max_pool(inputs, ksize=(2,2), strides=(2, 2), padding='VALID')

def conv_second_stage(x, ev):
    subset = np.expand_dims(x[...,0], -1)
    ans_CONV2 = conv2d(subset,ev)
    for i in range(4):
        subset = np.expand_dims(x[...,i +1], -1)
        conv = conv2d(subset,ev)
        ans_CONV2 = np.concatenate([ans_CONV2, conv], axis=-1)
    return ans_CONV2


class Filter():
    
    def __init__(self, amount, size):
        self.amount = amount
        self.size = size
        
class PCA_Preprocessing_Model():

    def __init__(self):
        self.pc_b1 = np.array([])
        self.pc_b2 = np.array([])
        
    def fit(self, data, filters_b1=Filter(5,3), filters_b2=Filter(10,8), experiment=0):
        self.pca = PCATransformer(experiment)
        self.pc_b1 = self.pca.extract_filters(data, filters_b1)
        print(self.pc_b1.shape)
        x = self.pca.transform(data,self.pc_b1)
        self.pc_b2 = self.pca.extract_filters(x, filters_b2)
        x = maxpool(x)
        x = self.pca.transform(x,self.pc_b2)
        x = maxpool(x)
        x = Flatten()(x)
        
        return x 
    
    def preprocess(self, data):
        x = self.pca.transform(data, self.pc_b1)
        x = maxpool(x)
        x = self.pca.transform(x, self.pc_b2)
        x = maxpool(x)
        x = Flatten()(x)
        return x

class PCATransformer():
    def __init__(self, experiment=0):
        self.experiment=experiment

    def extract_filters(self, data, filters):
        stage= 1 if data.shape[-1] == 1 else 2
        pklname_filters = f'ev_stage_{stage}_experiment_{self.experiment}.pkl'
        if exists(os.path.join(pkls_dir,pklname_filters)):
            return joblib.load(os.path.join(pkls_dir,pklname_filters))
        print('stage ', stage, 'data shape: ', data.shape)
        matrix = np.empty((filters.size ** 2, data.shape[2]**2))
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                # patchify
                current_im = data[i]
                patched_image = self.__patchify(np.expand_dims(current_im[..., j], -1), filters.size)
                # stacked matrix
                reshaped_patched_image = self.__create_stacked_matrix(patched_image)
                # res 9, 4096
                matrix = np.concatenate((matrix, reshaped_patched_image), axis=1)
        
        # cov matrix
        cov_m = np.cov(matrix)
        # evs 
        eigenValues, eigenVectors = lg.eig(cov_m)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        eigenvector_subset = eigenVectors[:filters.amount]
        eigenvector_subset_reshaped = eigenvector_subset.reshape(filters.amount, filters.size, filters.size)
        eigenvector_subset_reshaped = np.expand_dims(eigenvector_subset_reshaped, -1)
        eigenvector_subset_reshaped = eigenvector_subset_reshaped.transpose(1,2,3,0)
        # save experiment EVs
        print('dumping: ', os.path.join(pkls_dir,pklname_filters))
        joblib.dump(eigenvector_subset_reshaped, os.path.join(pkls_dir,pklname_filters))
        
        return eigenvector_subset_reshaped
    
    def __create_stacked_matrix(self, image):
        # e.g. input shape = 1, 64, 64, 9 --> output 9, 4096
        
        stacked_matrix=np.reshape(np.ones(image.shape[1]*image.shape[2]*image.shape[3]), (image.shape[3], image.shape[1]*image.shape[2]))
        for i in range(image.shape[1]):
            for j in range(image.shape[1]):
                patch = image[0,i,j,]
                mean_patch = patch - np.mean(patch)
                stacked_matrix[:,i*image.shape[1] + j] = mean_patch.T
        return stacked_matrix
    
    
    def __convolution2d(self, data, filters):
        
      
        if data.shape[-1] == 1:
            return tf.nn.conv2d(data, filters , strides=[1, 1, 1, 1] , padding='VALID' ) 
        
        else:
            subset = np.expand_dims(data[...,0], -1)
            res = conv2d(subset,filters)
            for i in range(data.shape[-1] - 1):
                subset = np.expand_dims(data[...,i +1], -1)
                conv = conv2d(subset,filters)
                res = np.concatenate([res, conv], axis=-1)
        return res
     
    def transform(self, data, filters):
        return self.__convolution2d(data, filters)
    
    def __patchify(self, img, patch_shape):
#         assert len(img.shape) == 4
        im_to_tensor = tf.convert_to_tensor(img)
        patched_tf = tf.image.extract_patches(images=im_to_tensor,sizes=[1, patch_shape, patch_shape, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1],padding='SAME')
        return patched_tf.numpy()
    
class PCA_Noise_Type_Classifier():
    
    def __init__(self, filters_b1=Filter(5,3), filters_b2=Filter(10,8), hidden_layers=(144, 3), activation_function='logistic'):
        self.pc_b1 = filters_b1
        self.pc_b2 = filters_b2
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_function, solver='adam', random_state=1,  max_iter=30)
        self.pca = PCA_Preprocessing_Model()
        
    def fit(self, x_train, y_train):
        preprocessed = self.pca.fit(x_train, self.pc_b1, self.pc_b2)
        X_df_train = pd.DataFrame()
        for tensor in preprocessed:
            X_df_train = X_df_train.append([tensor.numpy()])
        self.classifier.fit(X_df_train, y_train)
        
        
    def predict(self, x):
        preprocessed = self.pca.preprocess(x)
        X_df = pd.DataFrame()
        for tensor in preprocessed:
            X_df = X_df.append([tensor.numpy()])
        print()
        return self.classifier.predict(X_df), self.classifier.predict_proba(X_df)
        
    def get_score(self, x, y):
        preprocessed = self.pca.preprocess(x)
        X_df = pd.DataFrame()
        for tensor in preprocessed:
            X_df = X_df.append([tensor.numpy()])
        return self.classifier.score(X_df, y)
        
        
    def show_confusion_matrix(self, y_actual, y_predicted, labels_list=[]):
        df_preds_type = pd.DataFrame({'Actual noise': y_actual, 'Predicted noise': y_predicted})
        cm = confusion_matrix(y_actual.astype(int), y_predicted)
        fig, ax = plt.subplots(figsize=(6,5)) 
        ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        ax.set_title('Confusion Matrix\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        if len(labels_list) == 0:
            labels_list = ['Gaussian', 'Salt & Pepper', 'Speckle','Clean'] if np.unique(y_actual).size==4 else np.arange(start=0, stop=np.unique(y_actual).size) 
        ax.xaxis.set_ticklabels(labels_list)
        ax.yaxis.set_ticklabels(labels_list)
        plt.show()
        
class PCA_Noise_Level_Regressor():
    
    def __init__(self, filters_b1=Filter(5,3), filters_b2=Filter(10,8), hidden_layers=(144, 3), activation_function='logistic'):
        self.pc_b1 = filters_b1
        self.pc_b2 = filters_b2
        self.regressor = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation_function, verbose=1, solver='adam', batch_size=4, max_iter=30)
        self.pca = PCA_Preprocessing_Model()
        
    def fit(self, x_train_images, x_train_noise_type, y_train):
        preprocessed = self.pca.fit(x_train_images, self.pc_b1, self.pc_b2)
        X_df_train = pd.DataFrame()
        for tensor in preprocessed:
            X_df_train = X_df_train.append([tensor.numpy()])
        X_df_train['noise_type'] = x_train_noise_type
        self.regressor.fit(X_df_train, y_train)
        
    def predict(self, x_images, x_noise_type):
        preprocessed = self.pca.preprocess(x_images)
        X_df = pd.DataFrame()
        for tensor in preprocessed:
            X_df = X_df.append([tensor.numpy()])
        X_df['noise_type'] = x_noise_type
        
        return self.regressor.predict(X_df)
        
    def get_score(self, x_images, x_noise_type, y):
        preprocessed = self.pca.preprocess(x_images)
        X_df = pd.DataFrame()
        for tensor in preprocessed:
            X_df = X_df.append([tensor.numpy()])
        X_df['noise_type'] = x_noise_type
        score = self.regressor.score(X_df, y)
        
        return score
        
    def show_loss_curve(self):
        plt.plot(self.regressor.loss_curve_)
        plt.title("Loss Curve", fontsize=14)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
        
        
    def show_metrics(self, y_actual, y_predicted, labels_list=[]):
        print('Mean Absolute Error:', metrics.mean_absolute_error( y_actual, y_predicted))  
        print('Mean Squared Error:', metrics.mean_squared_error( y_actual, y_predicted))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_actual, y_predicted)))
    
    def results_head(self, predicted, actual, preview_amount=30):
        df_predictions = pd.DataFrame({'Actual':actual, 'Predicted': predicted})
        df_temp = df_predictions.head(preview_amount)
        df_temp.plot(kind='bar',figsize=(10,6))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()        
    