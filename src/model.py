import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from sklearn.metrics import ConfusionMatrixDisplay
import pathlib
import matplotlib.pyplot as plt


try:
    AUTOTUNE = tf.data.AUTOTUNE
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

num_classes = 6


class ModelHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.epochs = 70
        self.batch_size = 32
        self.train_ds = self.load_dataset_from_directory("train")
        self.valid_ds = self.load_dataset_from_directory("valid")
        self.test_ds = self.load_dataset_from_directory('test')
        self.model = self.create_model()
        self.compile_model()
        self.history = self.fit_model(self.epochs)

    def load_dataset_from_directory(self, dir_name: str) -> tf.data.Dataset:
        return tf.keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/{dir_name}",
            seed=123,
            image_size=(128, 128),
            batch_size=self.batch_size
        )

    @staticmethod
    def create_model() -> Sequential:
        normalization_layer = Rescaling(1. / 255)
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.15),
            RandomZoom(0.3)
        ])
        return Sequential([
            Input(shape=(128, 128, 3)),
            normalization_layer,
            data_augmentation,
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            Dropout(0.2),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.35),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes)
        ])

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def show_model_summary(self):
        self.model.summary()

    def fit_model(self, epochs: int):
        # self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # self.valid_ds = self.valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.train_ds = self.train_ds.shuffle(100, reshuffle_each_iteration=True)
        self.valid_ds = self.valid_ds.shuffle(100, reshuffle_each_iteration=True)
        return self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=epochs
        )

    def show_accuracy_plots(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

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

    def evaluate(self):
        score = self.model.evaluate(self.test_ds, batch_size=self.batch_size, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save_model(self):
        tf.keras.models.save_model(self.model, f"{os.getcwd()}/models/chess_v{'x'}")

    def plot_predict(self):
        fig, ax = plt.subplots(nrows=3, ncols=3)
        fig.tight_layout(h_pad=3)
        labels = os.listdir(f"{self.dataset_path}/test")
        predictions = {}
        for image_batch, label_batch in self.test_ds.take(1).as_numpy_iterator():
            predict = self.model.predict(image_batch)
            predicted_labels = []
            for p in predict:
                predicted_labels.append(np.minimum(5, np.argmax(p)))
            predicted_labels = np.array(predicted_labels)
            for i, image in enumerate(image_batch):
                index = predicted_labels[i]
                predictions[i] = (image / 255, labels[index])
                if i == 9:
                    break
            break
        idx = 0
        for i in range(3):
            for j in range(3):
                ax[i][j].imshow(predictions[idx][0])
                ax[i][j].set_title(predictions[idx][1], fontsize=12)
                idx += 1
        plt.show()
        fig.savefig('prediction.png')

    def show_conf_matrix(self):
        predictions = self.model.predict(self.test_ds.as_numpy_iterator())
        predicted_labels = []
        for p in predictions:
            predicted_labels.append(np.minimum(5, np.argmax(p)))
        predicted_labels = np.array(predicted_labels)
        lst = []
        for image_batch, label_batch in self.test_ds.as_numpy_iterator():
            for label in label_batch:
                lst.append(np.minimum(5, label))
        lst = np.array(lst)
        ConfusionMatrixDisplay(np.array(tf.math.confusion_matrix(lst,
                                                                 predicted_labels, num_classes=6))).plot()
        plt.show()
