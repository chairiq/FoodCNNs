# -*- coding: utf-8 -*-
"""SpringBoChap_PureFoodNet.ipynb

## PureFoodNet implementation
"""

#libraries
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

K.clear_session()

class PureFoodNet:
    # The model
    def getModel(input_shape=(224,224,3), num_classes=3):
        
        model = Sequential()
                
        #Block 1
        model.add(Conv2D(input_shape = input_shape,
                         filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', name='block1_conv1',
                         activation ='relu', kernel_initializer='he_normal'))
        model.add(Conv2D(filters = 128, kernel_size = (5,5), strides = 2, padding = 'Same', name='block1_conv2',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(MaxPool2D(strides=(2, 2), name='block1_pool'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        #Block 2
        model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv1',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv2',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', name='block2_conv3',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(MaxPool2D(strides=(2, 2), name='block2_pool'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        
        #Block 3
        model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv1',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv2',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', name='block3_conv3',
                         activation ='relu',kernel_initializer='he_normal'))
        model.add(MaxPool2D(strides=(2, 2), name='block3_pool'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        
        #Block 4
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation = "relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes,
                        activation = "softmax",
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2()))

        return model

img_width, img_height = 299, 299
train_data_dir = 'food-101/train/'
validation_data_dir = 'food-101/test/'
specific_classes = None #['apple_pie', 'greek_salad', 'baklava']
batch_size = 128

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=10,
    horizontal_flip=True,
    fill_mode='constant' 
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    classes = specific_classes,
    directory = train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    classes = specific_classes,
    directory = validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
n_classes = train_generator.num_classes

model_name = 'PureFoodNet_299x299Nadam_2'
epoch_num = 50

model = PureFoodNet.getModel(input_shape=train_generator.image_shape,
                                  num_classes = n_classes)
model.summary()

# learning rate scheduler
def schedule(epoch):
    if epoch < 10:
         new_lr = .001
    elif epoch < 14:
         new_lr = .0006
    elif epoch < 17:
         new_lr = .0003
    elif epoch < 20:
         new_lr = .0001
    elif epoch < 23:
         new_lr = .00005
    else:
         new_lr = .00001
    
    print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
    return new_lr
    
lr_scheduler = LearningRateScheduler(schedule)

model.compile(optimizer='Nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath='best_model_food101_'+model_name+'.hdf5',
                               verbose=1,
                               save_best_only=True)

csv_logger = CSVLogger('hist_food101_'+model_name+'.log')

hist = model.fit_generator(train_generator,
                           steps_per_epoch = nb_train_samples // batch_size,
                           validation_data = validation_generator,
                           validation_steps = nb_validation_samples // batch_size,
                           epochs = epoch_num,
                           verbose = 1,
                           callbacks = [csv_logger, checkpointer, lr_scheduler]
                          )

