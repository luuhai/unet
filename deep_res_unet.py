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


def deep_res_unet(pretrained_weights = None,input_size = (512,512,1), n_features=64):
    inputs = Input(input_size)
    conv1 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    plus1 = add([inputs, conv1])

    conv2 = BatchNormalization()(plus1)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(2*n_features, 3, strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    shortcut = Conv2D(2*n_features, 1, strides=(2, 2), padding='same')(plus1)
    shortcut = BatchNormalization()(shortcut)
    plus2 = add([shortcut, conv2])

    conv3 = BatchNormalization()(plus2)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(4*n_features, 3, strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(4*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    shortcut = Conv2D(4*n_features, 1, strides=(2, 2), padding='same')(plus2)
    shortcut = BatchNormalization()(shortcut)
    plus3 = add([shortcut, conv3])

    conv4 = BatchNormalization()(plus3)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(8*n_features, 3, strides=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(8*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    shortcut = Conv2D(8*n_features, 1, strides=(2, 2), padding='same')(plus3)
    shortcut = BatchNormalization()(shortcut)
    plus4 = add([shortcut, conv4])

    up8 = UpSampling2D(size = (2,2))(plus4)
    merge8 = concatenate([plus3,up8], axis = 3)
    conv8 = BatchNormalization()(merge8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(4*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(4*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    shortcut = Conv2D(4*n_features, 1, padding='same')(merge8)
    shortcut = BatchNormalization()(shortcut)
    plus8 = add([shortcut, conv8])

    up9 = UpSampling2D(size = (2,2))(plus8)
    merge9 = concatenate([plus2,up9], axis = 3)
    conv9 = BatchNormalization()(merge9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)
    conv9 = Conv2D(2*n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    shortcut = Conv2D(2*n_features, 1, padding='same')(merge9)
    shortcut = BatchNormalization()(shortcut)
    plus9 = add([shortcut, conv9])

    up10 = UpSampling2D(size = (2,2))(plus9)
    merge10 = concatenate([plus1,up10], axis = 3)
    conv10 = BatchNormalization()(merge10)
    conv10 = ReLU()(conv10)
    conv10 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)
    conv10 = Conv2D(n_features, 3, padding = 'same', kernel_initializer = 'he_normal')(conv10)
    shortcut = Conv2D(n_features, 1, padding='same')(merge10)
    shortcut = BatchNormalization()(shortcut)
    plus10 = add([shortcut, conv10])


    # conv14 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(plus13)
    # conv14 = ReLU()(conv14)
    conv14 = Conv2D(1, 1, activation = 'sigmoid')(plus10)

    model = Model(input = inputs, output = conv14)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer = SGD(lr = 1e-5, decay = 1e-7), loss = 'mean_squared_error', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


