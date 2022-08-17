import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Layer, Activation, UpSampling2D,  \
    BatchNormalization, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, ReLU, Concatenate

# from https://github.com/developershutt/Autoencoders/blob/main/3%20-%20Denoise%20Autoencoder/Code.ipynb

def create_model(model_name):
    encoder_input = Input(shape=(256,256,1))
    x1 = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPool2D(pool_size = (2,2), padding='same')(x1)
    x2 = Conv2D(64, (3,3), activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPool2D(pool_size = (2,2), padding='same')(x2)
    x3 = Conv2D(32, (3,3), activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    encoded = MaxPool2D(pool_size = (2,2), padding='same')(x3)


    # Decoder
    x3 = Conv2D(32, (3,3), activation='relu', padding='same')(encoded)
    x3 = BatchNormalization()(x3)
    x3 = UpSampling2D((2,2))(x3)
    x2 = Conv2D(64, (3,3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = UpSampling2D((2,2))(x2)
    x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x2)
    x1 = BatchNormalization()(x1)
    x1 = UpSampling2D((2,2))(x1)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding= 'same')(x1)

    return Model(encoder_input, decoded)