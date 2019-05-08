from deep_unet import *
from unet import *
from data import *

import skimage.io as io
import skimage.transform as transform
import numpy as np
import shutil
import os

def morphology(img, kernal, mode):
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal[0], kernal[1]))
    if mode == 'erode':
        img = cv2.erode(img, structure, (-1, -1))
    elif mode == 'dilate':
        img = cv2.dilate(img, structure, (-1, -1))
    elif mode == 'close':
        img = cv2.dilate(img, structure, (-1, -1))
        img = cv2.erode(img, structure, (-1, -1))
    elif mode == 'open':
        img = cv2.erode(img, structure, (-1, -1))
        img = cv2.dilate(img, structure, (-1, -1))
    return img

model = deep_unet(n_features=64)
# model = unet()
model.load_weights('deep_unet_128_hw_40e.hdf5')

save_path = "results/current"
gt_path = "data/hw/test/label"
testGene = testGenerator("data/hw/test")
for img, img_path, img_shape, orig_img in testGene:
    img_basename = img_path.split("/")[-1]
    predict_basename = img_basename.split('.')[0] + '_predict.png'
    final_basename = img_basename.split('.')[0] + '_final.png'
    gt_basename = img_basename.split('.')[0] + '_gt.jpg'
    predict = model.predict(img)
    predict = predict[0][:, :, 0]
    predict = resizeToOrigShape(predict, img_shape)
    predict = morphology(predict, (25, 25), 'open')
    final = (predict > 0.5) * 255

    #final = np.ascontiguousarray(final, dtype=np.uint8)
    #_, conts, hierarchy = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #list_conts = []
    #for cnt in conts:
    #    x, y, w, h = cv2.boundingRect(cnt)
    #    list_conts.append((x, y, w, h))

    #final[:, :] = 0
    #for c in list_conts:
    #    x, y, w, h = c
    #    cv2.rectangle(final, (x, y), (x + w, y + h), 255, -1)

    io.imsave(os.path.join(save_path, img_basename), orig_img)
    io.imsave(os.path.join(save_path, predict_basename), predict)
    io.imsave(os.path.join(save_path, final_basename), final)
    shutil.copyfile(os.path.join(gt_path, img_basename), os.path.join(save_path, gt_basename))
