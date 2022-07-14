import numpy as np
import glob

from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
import os
import cv2
from pathlib import Path
import imblearn


image_size = 224




pre = keras.applications.vgg16.VGG16(include_top = False, input_shape=(image_size, image_size, 3))

for layer in pre.layers[:-4]:
  layer.trainable = False
  #print(layer)

model = keras.Sequential()
model.add(pre)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-6),
              metrics=['acc'])






train_dir =Path("train")

val_dir = Path("val")


def prepare_and_load(dir):

    normal_dir=dir/'NORMAL'
    pneumonia_dir=dir/'PNEUMONIA'

    normal_cases = normal_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_dir.glob('*.jpeg')
    data,labels=([] for x in range(2))
    def prepare(case):
        for img in case:
            img = cv2.imread(str(img))
            img = cv2.resize(img, (224,224))
            if img.shape[2] ==1:
                 img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if case==normal_cases:
                label = 0
            else:
                label = 1
            data.append(img)
            labels.append(label)
        return data,labels
    prepare(normal_cases)
    d,l=prepare(pneumonia_cases)
    d=np.array(d)
    l=np.array(l)
    return d,l

train_d, train_l = prepare_and_load(train_dir)

val_d, val_l = prepare_and_load(val_dir)

ros = imblearn.over_sampling.SMOTE(random_state=0)

train_d = train_d.reshape(len(train_d), image_size * image_size * 3)

train_x_resampled, train_y_resampled = ros.fit_resample(train_d, train_l)

train_x_resampled = train_x_resampled.reshape(-1, image_size, image_size, 3)

history = model.fit(
        x = train_x_resampled,
        y = train_y_resampled,
        epochs=50,
        validation_data=(val_d, val_l))


model.save('MyModel.h5')


def visualize_results(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()




visualize_results(history)
