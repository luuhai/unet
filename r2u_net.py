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

def recurrent_block(x, ch_out, t=2):
    for i in range(t):
        if i == 0:
            x1 = Conv2D(ch_out, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
            x1 = BatchNormalization()(x1)
            x1 = ReLU()(x1)
        x1 = add([x, x1])
        x1 = Conv2D(ch_out, 3, padding = 'same', kernel_initializer = 'he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)

    return x1

def rcnn_block(x, ch_out, t=2):
    x = Conv2D(ch_out, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x1 = recurrent_block(x, ch_out)
    x1 = recurrent_block(x1, ch_out)
    return add([x, x1])

def up_conv(x, ch_out):
    up = UpSampling2D(size = (2,2))(x)
    x1 = Conv2D(ch_out, 3, padding = 'same', kernel_initializer = 'he_normal')(up)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    return x1

def r2u_net(pretrained_weights = None,input_size = (512,512,1), n_features=64):
    inputs = Input(input_size)

    x1 = rcnn_block(inputs, n_features)

    x2 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = rcnn_block(x2, 2*n_features)

    x3 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = rcnn_block(x3, 4*n_features)

    x4 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = rcnn_block(x4, 8*n_features)

    x5 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = rcnn_block(x5, 16*n_features)

    d5 = up_conv(x5, 8*n_features)
    d5 = concatenate([x4,d5], axis = 3)
    d5 = rcnn_block(d5, 8*n_features)

    d4 = up_conv(d5, 4*n_features)
    d4 = concatenate([x3,d4], axis = 3)
    d4 = rcnn_block(d4, 4*n_features)

    d3 = up_conv(d4, 2*n_features)
    d3 = concatenate([x2,d3], axis = 3)
    d3 = rcnn_block(d3, 2*n_features)

    d2 = up_conv(d3, n_features)
    d2 = concatenate([x1,d2], axis = 3)
    d2 = rcnn_block(d2, n_features)

    # conv14 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(plus10)
    # conv14 = ReLU()(conv14)
    d1 = Conv2D(1, 1, activation = 'sigmoid')(d2)

    model = Model(input = inputs, output = d1)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer = Adam(lr = 1e-5, decay = 1e-7), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


