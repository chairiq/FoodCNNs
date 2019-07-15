# -*- coding: utf-8 -*-
"""SpringBoChap_Transfer_Learning.ipynb

### Import all necessary libraries
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3,VGG16,ResNet50,MobileNetV2, NASNetMobile
from tensorflow.keras.applications import NASNetLarge, InceptionResNetV2, DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np

# The following line imports the SimpleShallowNet, which is a shallow CNN
# developed for the purposes of the this book chapter
#from ipynb.fs.full.BCh_PureFoodNet import PureFoodNet
K.clear_session()

"""### Choose the model
#### Choose the model that you want to use by setting the value of the "use_the_model" variable from 1 to 8. We should highlight that models from 1 to 7, are popular pretrained networks with ImageNet dataset , which not include the top layers. The 8th model is a simple shallow CNN netword developed for the purposes of this book chapter and it is not pretrained.
"""

use_the_model = 9
model_name = ''

if use_the_model is 1:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model_name = 'InceptionV3'
    epoch_num = 50
    
elif use_the_model is 2: 
    base_model = VGG16(weights='imagenet', include_top=False)
    model_name = 'VGG16'
    epoch_num = 70
    
elif use_the_model is 3: 
    base_model = ResNet50(weights='imagenet', include_top=False)
    model_name = 'ResNet50'
    epoch_num = 30
    
elif use_the_model is 4: 
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    model_name = 'InceptionResNetV2'
    epoch_num = 50
    
elif use_the_model is 5: 
    base_model = NASNetMobile(input_shape=(224,224,3), weights='imagenet', include_top=False)
    model_name = 'NASNetMobile'
    epoch_num = 50
elif use_the_model is 6: 
    base_model = NASNetLarge(input_shape=(331,331,3), weights='imagenet', include_top=False)
    model_name = 'NASNetLarge'
    epoch_num = 50
    
elif use_the_model is 7: 
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model_name = 'MobileNetV2'
    epoch_num = 70
    
elif use_the_model is 8: 
    base_model = DenseNet121(weights='imagenet', include_top=False)
    model_name = 'DenseNet121'
    epoch_num = 50
    
elif use_the_model is 9: 
    base_model = PureFoodNet.getModel(input_shape=train_generator.image_shape)
    model_name = 'PureFoodNet'
    epoch_num = 300

print("({}) {} model loaded with {} epochs.".format(model_name,use_the_model, epoch_num))

"""### Prepare the training and the validation sets of the food101 dataset
#### Add a small image augmentation to the training set (shear_range, zoom_range, horizontal_flip)
"""

img_width, img_height = 299, 299
train_data_dir = 'food-101/train/'
validation_data_dir = 'food-101/test/'
batch_size = 256

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
n_classes = train_generator.num_classes

"""### Add new top layers to the selected model"""

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n_classes,
                    kernel_regularizer=regularizers.l2(0.005), 
                    activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

"""### Compile the model
#### Compile the model with SGD optimazer, and use top 1 and top 5 accuracy metrics. Initialize two callbacks, one for checkpoints and one for the training logs
"""

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath='best_model_food101_'+model_name+'.hdf5',
                               verbose=1,
                               save_best_only=True)
csv_logger = CSVLogger('hist_food101_'+model_name+'.log')

"""### Training session of the selected model"""

hist = model.fit_generator(train_generator,
                           steps_per_epoch = nb_train_samples // batch_size,
                           validation_data = validation_generator,
                           validation_steps = nb_validation_samples // batch_size,
                           epochs = epoch_num,
                           verbose = 1,
                           callbacks = [csv_logger, checkpointer]
                          )

"""### Save the last trained model"""

model.save('last_model_food101_'+str(model_name)+'_acc'+str(max(hist.history['acc']))+'.hdf5')
