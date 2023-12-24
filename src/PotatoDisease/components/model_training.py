import tensorflow as tf
from src.PotatoDisease.logging import logger
import mlflow
from urllib.parse import urlparse


class ModelTraining:
    def __init__(self, config):
        self.config = config

    def split_dataset(self,
                      dataset,
                      train_split=0.8,
                      test_split=0.1,
                      val_split=0.1,
                      shuffle=True):

        assert (train_split+test_split+val_split) == 1

        dataset_size = len(dataset)
        if shuffle:
            dataset = dataset.shuffle(10000, seed=42)

        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)

        train_data = dataset.take(train_size)
        val_data = dataset.skip(train_size).take(val_size)
        test_data = dataset.skip(train_size).skip(val_size)

        return train_data, val_data, test_data

    def prepare_dataset(self):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.dataset_path,
            image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        logger.info(f'name of classes: {dataset.class_names}')

        train_data, val_data, test_data = self.split_dataset(dataset)

        train_data = train_data.cache().shuffle(
            1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_data = val_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data = test_data.cache().shuffle(
            1000).prefetch(buffer_size=tf.data.AUTOTUNE)

        # applay data agumentation to train dataset
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        train_data = train_data.map(lambda x, y:
                                    (data_augmentation(x, training=True), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_data, val_data, test_data

    def training(self):
        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(
                self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        ])

        # create a model architecture
        input_shape = (self.config.BATCH_SIZE, self.config.IMAGE_SIZE,
                       self.config.IMAGE_SIZE, self.config.CHANNELS)
        n_classes = 3

        model = tf.keras.models.Sequential([
            resize_and_rescale,
            tf.keras.layers.Conv2D(32, kernel_size=(
                3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax'),
        ])
        model.build(input_shape=input_shape)
        logger.info(model.summary())

        # compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            metrics=['accuracy']
        )

        train_data, val_data, test_data = self.prepare_dataset()

        mlflow.set_experiment('Tensorflow Models')
        mlflow.set_registry_uri(
            'https://dagshub.com/fraidoon_omarzai/Potato-disease-end-to-end.mlflow')

        with mlflow.start_run(run_name="Tf") as mlops_run:

            # train the model
            model.fit(
                train_data,
                batch_size=self.config.BATCH_SIZE,
                validation_data=val_data,
                verbose=1,
                epochs=self.config.EPOCHS
            )

            # evaluate the model
            results = model.evaluate(test_data)
            logger.info(f"Model evaluation: {results}")

            mlflow.log_param('batch size', self.config.BATCH_SIZE)
            mlflow.log_param('epochs', self.config.EPOCHS)

            mlflow.log_metric('accuracy', results[1])

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.tensorflow.log_model(
                    model, "model", registered_model_name="TFModel")
            else:
                mlflow.tensorflow.log_model(model, "model")

    #     self.save_model(
    #         self.config.model_save,
    #         model
    #     )

    # @staticmethod
    # def save_model(path: Path, model: tf.keras.Model):
    #     model.save(path)
