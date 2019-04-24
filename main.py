from deep_unet import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/hw/train','image','label',data_gen_args,save_to_dir = None)
valGene = trainGenerator(2, 'data/hw/test', 'image', 'label', None, save_to_dir=None)

model = deep_unet()
model.summary()
model_checkpoint = ModelCheckpoint('unet_hw.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint], validation_data=valGene, validation_steps=1)

testGene = testGenerator("data/hw/test")
results = model.predict_generator(testGene,10,verbose=1)
saveResult("data/hw/test",results)
