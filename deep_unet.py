import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def deep_unet(pretrained_weights = None,input_size = (512,512,1), n_features=64):
    inputs = Input(input_size)
    conv1 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    plus2 = add([pool1, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(plus2)

    conv3 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    plus3 = add([pool2, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(plus3)

    conv4 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    plus4 = add([pool3, conv4])
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(plus4)

    conv5 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    plus5 = add([pool4, conv5])
    # drop4 = Dropout(0.5)(conv4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(plus5)

    conv6 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU()(conv6)
    plus6 = add([pool5, conv6])
    # drop4 = Dropout(0.5)(conv4)
    pool6 = MaxPooling2D(pool_size=(2, 2))(plus6)

    conv7 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool6)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU()(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    up8 = LeakyReLU()(up8)
    merge8 = concatenate([plus6,up8], axis = 3)
    conv8 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU()(conv8)

    up9 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = LeakyReLU()(up9)
    merge9 = concatenate([plus5,up9], axis = 3)
    conv9 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU()(conv9)

    up10 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    up10 = LeakyReLU()(up10)
    merge10 = concatenate([plus4,up10], axis = 3)
    conv10 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = LeakyReLU()(conv10)
    conv10 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv10 = LeakyReLU()(conv10)

    up11 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    up11 = LeakyReLU()(up11)
    merge11 = concatenate([plus3,up11], axis = 3)
    conv11 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = LeakyReLU()(conv11)
    conv11 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv11 = LeakyReLU()(conv11)

    up12 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv11))
    up12 = LeakyReLU()(up12)
    merge12 = concatenate([plus2,up12], axis = 3)
    conv12 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge12)
    conv12 = LeakyReLU()(conv12)
    conv12 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv12 = LeakyReLU()(conv12)

    up13 = Conv2D(n_features, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv12))
    up13 = LeakyReLU()(up13)
    merge13 = concatenate([conv1,up13], axis = 3)
    conv13 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge13)
    conv13 = LeakyReLU()(conv13)
    conv13 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv13)
    conv13 = LeakyReLU()(conv13)

    conv13 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv13)
    conv13 = LeakyReLU()(conv13)
    conv14 = Conv2D(1, 1, activation = 'sigmoid')(conv13)

    model = Model(input = inputs, output = conv14)

    model.compile(optimizer = Adam(lr = 4e-4, decay = 1e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


