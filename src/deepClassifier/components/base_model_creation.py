from pathlib import Path
from deepClassifier.entity import PrepareBaseModelConfig
import tensorflow as tf
from deepClassifier import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        # Load the VGG16 base model without including the top (fully connected) layers
        base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=False
        )

        # Freeze all layers of the base model
        base_model.trainable = False

        # Create a new model by adding custom layers for classification
        flatten_layer = tf.keras.layers.Flatten()(base_model.output)
        prediction_layer = tf.keras.layers.Dense(
            units=self.config.params_classes,
            activation="softmax"
        )(flatten_layer)

        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        full_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        # Prepare the full model by adding custom layers and compiling it
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated base model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
