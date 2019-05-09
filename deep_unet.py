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
from keras.utils.vis_utils import plot_model


def deep_unet(pretrained_weights = None,input_size = (512,512,1), n_features=64):
    inputs = Input(input_size)
    conv1 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    plus1 = add([inputs, conv1])
    conv1 = LeakyReLU()(plus1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    plus2 = add([pool1, conv2])
    conv2 = LeakyReLU()(plus2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    plus3 = add([pool2, conv3])
    conv3 = LeakyReLU()(plus3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    plus4 = add([pool3, conv4])
    conv4 = LeakyReLU()(plus4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    plus5 = add([pool4, conv5])
    conv5 = LeakyReLU()(plus5)
    # drop4 = Dropout(0.5)(conv4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    plus6 = add([pool5, conv6])
    conv6 = LeakyReLU()(plus6)
    # drop4 = Dropout(0.5)(conv4)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(pool6)
    conv7 = LeakyReLU()(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = UpSampling2D(size = (2,2))(drop7)
    merge8 = concatenate([plus6,up8], axis = 3)
    conv8 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU()(conv8)
    plus8 = add([up8, conv8])

    up9 = UpSampling2D(size = (2,2))(plus8)
    merge9 = concatenate([plus5,up9], axis = 3)
    conv9 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU()(conv9)
    plus9 = add([up9, conv9])
    # conv9 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = LeakyReLU()(conv9)

    up10 = UpSampling2D(size = (2,2))(plus9)
    merge10 = concatenate([plus4,up10], axis = 3)
    conv10 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = LeakyReLU()(conv10)
    conv10 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv10 = LeakyReLU()(conv10)
    plus10 = add([up10, conv10])
    # conv10 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv10)
    # conv10 = LeakyReLU()(conv10)

    up11 = UpSampling2D(size = (2,2))(plus10)
    merge11 = concatenate([plus3,up11], axis = 3)
    conv11 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = LeakyReLU()(conv11)
    conv11 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv11 = LeakyReLU()(conv11)
    plus11 = add([up11, conv11])
    # conv11 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv11)
    # conv11 = LeakyReLU()(conv11)

    up12 = UpSampling2D(size = (2,2))(plus11)
    merge12 = concatenate([plus2,up12], axis = 3)
    conv12 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge12)
    conv12 = LeakyReLU()(conv12)
    conv12 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv12 = LeakyReLU()(conv12)
    plus12 = add([up12, conv12])
    # conv12 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv12)
    # conv12 = LeakyReLU()(conv12)

    up13 = UpSampling2D(size = (2,2))(plus12)
    merge13 = concatenate([plus1,up13], axis = 3)
    conv13 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(merge13)
    conv13 = LeakyReLU()(conv13)
    conv13 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv13)
    conv13 = LeakyReLU()(conv13)
    plus13 = add([up13, conv13])
    # conv13 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv13)
    # conv13 = LeakyReLU()(conv13)

    conv14 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(plus13)
    conv14 = LeakyReLU()(conv14)
    conv14 = Conv2D(1, 1, activation = 'sigmoid')(conv14)

    model = Model(input = inputs, output = conv14)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer = Adam(lr = 1e-5, decay = 1e-7), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


