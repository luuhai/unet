from model import *
from data import *

import skimage.io as io
import skimage.transform as transform
import numpy as np
import shutil
import os

model = unet()
model.load_weights('unet_ffg2.hdf5')

save_path = "data/ffg/test"
gt_path = "data/ffg/test/label"
testGene = testGenerator("data/ffg/test")
for img, img_path, img_shape, orig_img in testGene:
    img_basename = img_path.split("/")[-1]
    predict_basename = img_basename.split('.')[0] + '_predict.png'
    final_basename = img_basename.split('.')[0] + '_final.png'
    gt_basename = img_basename.split('.')[0] + '_GT0.jpg'
    predict = model.predict(img)
    predict = predict[0][:, :, 0]
    predict = transform.resize(predict, img_shape)
    final = (predict > 0.2) * 255

    io.imsave(os.path.join(save_path, img_basename), orig_img)
    io.imsave(os.path.join(save_path, predict_basename), predict)
    io.imsave(os.path.join(save_path, final_basename), final)
    shutil.copyfile(os.path.join(gt_path, gt_basename), os.path.join(save_path, gt_basename))
