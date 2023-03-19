import tensorflow as tf
from functools import partial
from loss_metric_functions import customized_crossentropy, f1_score
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


L1 = 5e-4
L2 = 5e-5
conv2d_name_idx = 0
depth_name_idx = 0
bn_name_idx = 0
activation_name_idx = 0

def get_channel_axis():
    """
    Returns channel axis, 1 if channel first or -1 if last
    """
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    return channel_axis


def conv_block(inputs, filters, kernel, strides,  padding='same', trainable=True):
    global conv2d_name_idx, bn_name_idx, activation_name_idx
    """
    A basic Conv - BN - Activation block for reusability
    """
    channel_axis = get_channel_axis()
    x = tf.keras.layers.Conv2D(filters, kernel, padding=padding, strides=strides,
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2), trainable=trainable, name='conv2d_{}'.format(conv2d_name_idx))(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis, epsilon=1e-05, momentum=0.1, trainable=trainable, name='batch_{}'.format(bn_name_idx))(x)
    x = tf.keras.layers.ReLU(max_value=6, name='activation_{}'.format(activation_name_idx))(x)
    conv2d_name_idx += 1
    bn_name_idx += 1
    activation_name_idx += 1
    return x


def bottleneck(inputs, filters, kernel, expansion_factor, alpha_contraction, strides, r=False, trainable=True):
    """
    Mobilenet v2 bottleneck layer which has:
    1x1 convolutions for filter expansion
    kernel x kernel depthwise 2D convolutions
    1x1 convolutions for filter contraction
    :param inputs: an input layer, which is the last built layer in the model so far
    :param filters: number of filters
    :param kernel: kernel size
    :param expansion_factor: the expansion factor for the filters
    :param alpha_contraction: the percentage of the output channels
    :param strides: the strides for the Convolution layer
    :param r: whether to apply residual connection
    :param trainable: whether to make this block trainable
    :return: the bottleneck block
    """

    global conv2d_name_idx, bn_name_idx, activation_name_idx, depth_name_idx
    channel_axis = get_channel_axis()
    # expansion target
    expansion_channel = tf.keras.backend.int_shape(inputs)[channel_axis] * expansion_factor
    # contraction target
    contraction_channel = int(filters * alpha_contraction)

    x = conv_block(inputs, expansion_channel, (1, 1), (1, 1), trainable=trainable)
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(strides, strides), depth_multiplier=1,
                                        padding='same',
                                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2), trainable=trainable, name='depth_{}'.format(depth_name_idx))(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis, trainable=trainable, name='batch_{}'.format(bn_name_idx))(x)

    x = tf.keras.layers.ReLU(max_value=6, name='activation_{}'.format(activation_name_idx))(x)

    x = tf.keras.layers.Conv2D(contraction_channel, (1, 1), strides=(
        1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2), trainable=trainable, name='conv2d_{}'.format(conv2d_name_idx))(x)

    bn_name_idx += 1
    x = tf.keras.layers.BatchNormalization(axis=channel_axis, trainable=trainable, name='batch_{}'.format(bn_name_idx))(x)
    conv2d_name_idx += 1
    bn_name_idx += 1
    activation_name_idx += 1
    depth_name_idx += 1
    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x


def inverted_residual_block(inputs, filters, kernel, expansion_factor, alpha_contraction, strides, n, trainable=True):
    """
    Mobilenet v2 Inverted residual block
    :param inputs: an input layer, which is the last built layer in the model so far
    :param filters: number of filters
    :param kernel: kernel size
    :param expansion_factor: the expansion factor for the filters
    :param alpha_contraction: the percentage of the output channels
    :param strides: the strides for the Convolution layer
    :param n: how many times to apply the bottleneck block
    :param trainable: whether to make this block trainable
    :return: the final layer of this inverted residual block
    """
    x = bottleneck(inputs, filters, kernel, expansion_factor,
                        alpha_contraction, strides, False, trainable=trainable)

    for i in range(1, n):
        x = bottleneck(x, filters, kernel, expansion_factor, alpha_contraction, 1, True, trainable=trainable)

    return x


class NetworkX:
    def __init__(self,
                 input_shape,
                 feature_extractor=False,
                 classes=None,
                 model_saving_path='./',
                 first_filters=16,
                 last_filters=256,
                 alpha_contraction=1.0,
                 learning_rate=0.001,
                 pretrained_weights=None,
                 trainable=True

                 ):
        if not feature_extractor and classes is None:
            raise "Classes must not be None if feature extractor is False"

        self.classes = classes
        self.model_saving_path = model_saving_path
        self.input_shape = input_shape
        self.first_filters = first_filters
        self.last_filters = last_filters
        self.alpha_contraction = alpha_contraction
        self.learning_rate = learning_rate
        self.pretrained_weights = pretrained_weights
        self.trainable = trainable
        if not feature_extractor:
            self.model = self.build_model()
            self.compile_model()
            self.print_model()

    def build_feature_extractor(self):
        """
        Builds only the feature extractor part of this model, which is the CNN layers part
        :return: the input and the output layers
        """
        input_ = tf.keras.layers.Input(shape=self.input_shape)
        x = conv_block(input_, self.first_filters, (3, 3), strides=(2, 2), trainable=self.trainable)

        x = inverted_residual_block(x, 32, (3, 3), expansion_factor=1, alpha_contraction=self.alpha_contraction,
                                    strides=2, n=2, trainable=self.trainable)
        x = inverted_residual_block(x, 64, (3, 3), expansion_factor=1, alpha_contraction=self.alpha_contraction,
                                    strides=2, n=1, trainable=self.trainable)
        x = inverted_residual_block(x, 96, (3, 3), expansion_factor=1, alpha_contraction=self.alpha_contraction,
                                    strides=2, n=2, trainable=self.trainable)
        x = inverted_residual_block(x, 128, (3, 3), expansion_factor=1, alpha_contraction=self.alpha_contraction,
                                    strides=2, n=1, trainable=self.trainable)

        # expression subspace
        features = conv_block(x, self.last_filters, (1, 1), strides=(2, 2), trainable=self.trainable)
        features = tf.keras.layers.GlobalAveragePooling2D(name='global_avg')(features)
        return input_, features

    def build_model(self):
        """
        builds the model by calling the feature extractor and append to it a prediction layer
        """
        input_, features = self.build_feature_extractor()
        dropout = tf.keras.layers.Dropout(rate=0.3)(features)
        emotions_output = tf.keras.layers.Dense(len(self.classes), name='prediction')(dropout)
        emotions_activation = tf.keras.layers.Activation('sigmoid', name='prediction_activation')(emotions_output)
        model = tf.keras.models.Model(input_, [emotions_activation])

        # check if we need to run this model with pretrained weights
        if self.pretrained_weights:
            model.load_weights(self.pretrained_weights, by_name=False)
        return model

    def compile_model(self):
        """
        compiles the model with the needed paratmers
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
        prints the model's summary
        """
        self.model.summary()

    def train_network(self, training_data, validation_data, validation_batch_size):
        """
        trains the network
        :param training_data: training data generator
        :param validation_data: validation data generator
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

    def predict(self, data):
        return self.model.predict(data)

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
