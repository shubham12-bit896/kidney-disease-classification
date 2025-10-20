import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.cnnClassifier.entity.config_entity import TrainingConfig


# ...existing code...
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # load without loading optimizer state
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        # compile with a fresh optimizer (adjust loss/optimizer to your project)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        # ...existing code...
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode='sparse',
            **dataflow_kwargs,
        )
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode='sparse',
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # ensure optimizer is fresh (if you changed trainable flags, recreate optimizer and recompile)
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model,
        )