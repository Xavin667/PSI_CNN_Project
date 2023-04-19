```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from PIL import Image

# Open image file and get the dimensions
img = Image.open('test/Carrot/1001.jpg')
shape = img.size
print(f'Image size: {shape}')
# Define parameters
batch_size = 128
num_classes = 15
epochs = 5
steps = 100
input_shape = (shape[0], shape[1], 3)

# Paths to image directories
TRAIN_DIR = 'train'
TEST_DIR = 'test'
VALIDATION_DIR = 'validation'
```

![output_img_shape](/output_ss/img_shape.png "Image shape")

```python
# Data generator with no data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generator with data augmentation
train_datagen_aug = ImageDataGenerator(rescale=1./255, # Rescale pixel values to [0, 1]
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 

train_generator_aug = train_datagen_aug.flow_from_directory(
    TRAIN_DIR, target_size=shape, batch_size=batch_size, class_mode='categorical')

# Load data with data generators
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=shape, batch_size=batch_size, class_mode='categorical')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=shape, batch_size=batch_size, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=shape, batch_size=batch_size, class_mode='categorical')

# Shape of data
print(f'Data shape: {train_generator[0][0].shape}')

# Number of classes
print(f'Number of classes: {len(train_generator.class_indices)}')
```

![data](/output_ss/data.png "Data")

```python
#Shallow model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

print("Model parameters = %d" % model.count_params())
print(model.summary())

history = model.fit(train_generator, epochs=8, validation_data=validation_generator, validation_steps=steps)

score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

```python
# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze pre-trained layers (with them epoch takes 4 hours)
for layer in base_model.layers:
    layer.trainable = False
    
model = Sequential()
model.add(base_model)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

print("Model parameters = %d" % model.count_params())
print(model.summary())

# Train the model with no data augmentation
history = model.fit(train_generator, epochs=2, validation_data=validation_generator, validation_steps=steps)

score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Train the model with no data augmentation
history = model.fit( train_generator_aug, steps_per_epoch=steps, epochs=epochs,
    validation_data=validation_generator, validation_steps=steps)

score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

![vgg16_data](/output_ss/vgg16_data.png "VGG16 output data")

```python
# Deep model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

print("Model parameters = %d" % model.count_params())
print(model.summary())

# Train the model with no data augmentation
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Train the model with no data augmentation
history = model.fit( train_generator_aug, steps_per_epoch=steps, epochs=epochs,
    validation_data=validation_generator, validation_steps=steps)

score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
