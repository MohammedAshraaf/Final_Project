import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from functools import partial
from loss_metric_functions import customized_crossentropy, auc_roc_per_class_sklearn, f1_score


class MobilenetModel:
    def __init__(self, classes, input_shape, model_saving_path, pretrained_weights='', learning_rate=0.001):
        self.classes = classes
        self.input_shape = input_shape
        self.model_saving_path = model_saving_path
        self.learning_rate = learning_rate
        self.pretrained_weights = pretrained_weights
        self.model = self.build_model()
        self.compile_model()
        self.print_model()

    def build_model(self):
        """
        creates the model, whether mobilenet V2 or V3
        :return: the created model
        """
        input_ = tf.keras.Input(self.input_shape)
        convert_RGB = tf.keras.layers.Conv2D(3, (1, 1), padding='same', strides=1)(input_)
        model = tf.keras.applications.MobileNetV2(
            input_shape=[self.input_shape[0], self.input_shape[1], 3],
            alpha=0.35,
            include_top=False,
            weights=self.pretrained_weights,
            pooling='avg',
        )
        features = model(convert_RGB)
        dropout_layer = tf.keras.layers.Dropout(rate=0.5)(features)
        prediction_layer = tf.keras.layers.Dense(len(self.classes), activation='sigmoid')(dropout_layer)

        return tf.keras.Model(inputs=input_, outputs=prediction_layer)

    def compile_model(self):
        """
        compiles the model to be ready for training
        """
        metrics = self.get_metrics()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=customized_crossentropy,
            metrics=metrics,
            run_eagerly=True
        )

    def print_model(self):
        """
        prints model's summary
        """
        self.model.summary()

    def train_network(self, training_data, validation_data, validation_batch_size):
        """
        trains the network
        :param training_data: training data generator
        :param validation_data: validation data x, y
        :param validation_batch_size: batch size for validation data
        :return: history of the training
        """
        print("validation steps: ", validation_data[0].shape[0] // validation_batch_size)
        callbacks = self.get_callbacks()
        history = self.model.fit(
            training_data,
            steps_per_epoch=100,
            epochs=200,
            verbose=1,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_batch_size=validation_batch_size,
            validation_steps=validation_data[0].shape[0] // validation_batch_size,
            shuffle=True,
            initial_epoch=0,
        )
        return history

    def learning_rate_scheduler(self, epoch, lr):
        """
        applies learning rate decay based on the current epoch
        :param epoch: the current training epoch
        :param lr: the current learning rate
        :return: the updated learning rate
        """
        if epoch % 25 == 0 and epoch != 0:
            lr /= 2
        return lr

    def get_callbacks(self):
        """
        creates the needed callbacks for training
        :return: list of callbacks
        """
        LR_callback = tf.keras.callbacks.LearningRateScheduler(self.learning_rate_scheduler, verbose=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            '{}/best_weights.h5'.format(self.model_saving_path),
            verbose=1,
            monitor='val_f1',
            save_best_only=True,
            mode='max'
        )
        log_dir = self.model_saving_path + "/logs/"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return [LR_callback, checkpoint_callback, tensorboard_callback]


    def get_metrics(self):
        """
        gets the metrics for the training
        :return: list of the metrics
        """
        metrics = []
        func = partial(f1_score, cls=-1)
        func.__name__ = 'f1'
        metrics.append(func)
        return metrics