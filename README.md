# Image restoration using neural networks
Noise analysis and denoising using CNN and python.

This project consists in total of four networks. The first two cary out te task of noise analysis. They are PCA-based and classify the type of noise and estimate the noise variance based on the obtained knowledge. The second two handle the removal of the noise and present two different approaches employing autoencoders. The first one, GDAE, reconstructs clean images from degraded input without requiring any prior knowledge of the degradation function or its parameter-space. The second one, SDAE, is connected to the noise classifier and build together a toolchain, where for each type of noise there is an autoencoder, specialized in removing it.

## Structure and usage
Pretrained models are in folder checkpoints.

Download training data from: https://drive.google.com/drive/folders/1zRwVhb6osPEvbpoYSKqi6muRvGcz2rBG?usp=sharing
Dataset must be in folder data with following structure: \
├── data \
│   ├── custom_noises \
│   │   ├── clean \
│   │   │   ├── clean \
│   │   ├── gaussian \
│   │   │   ├── gaussian \
│   │   ├── sp \
│   │   │   ├── s&p \
│   │   ├── speckle \
│   │   │   ├── speckle 


In src folder there are two executable scripts for each model - training and predictions. Important is to execute scripts directly from the folder they are in! 
Scrips for training accept two optional boolean parameters. First one is "augmented", depending on which the training data is augmented and its size is increased three times. Due to possible hardware limitations its default value is false. The second one is "overwrite". Once the dataset is loaded from the local directory and the images are preprocessed, the results are saved to pikles, so loading them again is easier next time. If some parameters in the preprocessing are adjusted, this flag should be set to true, so the data is loaded again and the old pikles are replaced, otherwise the last version would be used again. Default value is false.

Scripts for prediction require two parameters, one for path of the input file, one for path of the output file. For testing purposes the Lena image in data folder can be used, several versions of it with different noise types and levels are uploaded. 




