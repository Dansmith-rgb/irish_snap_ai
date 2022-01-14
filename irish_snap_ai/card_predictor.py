from sklearn.utils import validation
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomRotation, RandomContrast
from sklearn.model_selection import train_test_split
import keras
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import lite

def load_data():
    image_size = (300, 200)
    batch_size = 32
    color_mode = "grayscale"
    train_ds = keras.preprocessing.image_dataset_from_directory(
        "images",
        validation_split=0.2,
        subset="training",
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        labels="inferred",
        seed=1337,
        shuffle=True
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        "images",
        validation_split=0.2,
        subset="validation",
        image_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=1337,
        shuffle=True,
    )

    return train_ds, val_ds

train_ds, val_ds = load_data()
class_names = train_ds.class_names

def create_model(class_names, train_ds, val_ds):
    num_classes = len(class_names)

    

    data_augmentation = keras.Sequential(
   [
    
    RandomRotation(0.1, input_shape=(300, 200, 1)),
    RandomContrast(0.5)
    #layers.RandomZoom(0.1),
   ]
  )
    #CNN Model
    model = models.Sequential()
    model.add(data_augmentation)
    model.add(Rescaling(1./255))
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu')),
    model.add(layers.Dense(num_classes))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    epochs=150
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    keras_file = "predictor.h5"
    keras.models.save_model(model, keras_file)

    keras_model = tf.keras.models.load_model("predictor.h5")
    converter = lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
    

#create_model(class_names, train_ds, val_ds)

def predict_card(model, img):
    img = img
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return class_names[np.argmax(score)]
    