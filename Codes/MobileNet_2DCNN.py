"""MobileNet 2DCNN in Keras.
MovileNet_v1: https://arxiv.org/abs/1704.04861 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]
MovileNet_v2: https://arxiv.org/abs/1801.04381 [Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation]
MovileNet_v3: https://arxiv.org/abs/1905.02244 [Searching for MobileNetV3]
"""


import tensorflow as tf


def Conv_2D_block(inputs, model_width, kernel, strides):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=(strides, strides), padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Conv_2D_DW(inputs, model_width, kernel, strides, alpha):
    # 2D Depthwise Separable Convolutional Block with BatchNormalization
    model_width = int(model_width * alpha)
    x = tf.keras.layers.SeparableConv2D(model_width, kernel, strides=(strides, strides), depth_multiplier=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(model_width, 1, strides=(1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def bottleneck_block(inputs, filters, kernel, t, alpha, strides, r=False):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t
    cchannel = int(filters * alpha)

    x = Conv_2D_block(inputs, tchannel, 1, 1)
    x = tf.keras.layers.SeparableConv2D(filters, kernel, strides=(strides, strides), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(cchannel, 1, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('linear')(x)

    if r:
        x = tf.keras.layers.concatenate([x, inputs], axis=-1)

    return x


def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    if strides == 1:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides, True)
    else:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = bottleneck_block(x, filters, kernel, t, alpha, 1, True)

    return x


def Conv_2D_block_2(inputs, model_width, kernel, strides, nl):
    # This function defines a 2D convolution operation with BN and activation.
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=(strides, strides), padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    if nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    return x


def _squeeze(inputs):
    # This function defines a squeeze structure.

    input_channels = int(inputs.shape[-1])

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(input_channels, activation='relu')(x)
    x = tf.keras.layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = tf.keras.layers.Reshape((1, input_channels))(x)
    x = tf.keras.layers.Multiply()([inputs, x])

    return x


def bottleneck_block_2(inputs, filters, kernel, e, s, squeeze, nl, alpha):
    # This function defines a basic bottleneck structure.

    input_shape = tf.keras.backend.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[2] == filters

    x = Conv_2D_block_2(inputs, tchannel, 1, 1, nl)

    x = tf.keras.layers.SeparableConv2D(filters, kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    if nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    if squeeze:
        x = _squeeze(x)

    x = tf.keras.layers.Conv2D(cchannel, 1, strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x


class MobileNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, alpha=1.0):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.alpha = alpha

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPool2D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def MobileNet_v1(self):
        inputs = tf.keras.Input((self.length, self.num_channel))

        x = Conv_2D_block(inputs, self.num_filters * (2 ** 0), 3, 2)
        x = Conv_2D_DW(x, self.num_filters, 3, 1, self.alpha)
        x = Conv_2D_DW(x, self.num_filters * (2 ** 1), 3, 2, self.alpha)
        x = Conv_2D_DW(x, self.num_filters, 3, 1, self.alpha)
        x = Conv_2D_DW(x, self.num_filters * (2 ** 2), 3, 2, self.alpha)
        x = Conv_2D_DW(x, self.num_filters, 3, 1, self.alpha)
        x = Conv_2D_DW(x, self.num_filters * (2 ** 3), 3, 2, self.alpha)
        for i in range(5):
            x = Conv_2D_DW(x, self.num_filters, 3, 1, self.alpha)
        x = Conv_2D_DW(x, self.num_filters * (2 ** 4), 3, 2, self.alpha)
        x = Conv_2D_DW(x, self.num_filters * (2 ** 5), 3, 2, self.alpha)

        outputs = self.MLP(x)
        model = tf.keras.Model(inputs, outputs)

        return model

    def MobileNet_v2(self):
        inputs = tf.keras.Input((self.length, self.num_channel))
        x = Conv_2D_block(inputs, self.num_filters, 3, 2)

        x = inverted_residual_block(x, 16, 3, t=1, alpha=self.alpha, strides=1, n=1)
        x = inverted_residual_block(x, 24, 3, t=6, alpha=self.alpha, strides=2, n=2)
        x = inverted_residual_block(x, 32, 3, t=6, alpha=self.alpha, strides=2, n=3)
        x = inverted_residual_block(x, 64, 3, t=6, alpha=self.alpha, strides=2, n=4)
        x = inverted_residual_block(x, 96, 3, t=6, alpha=self.alpha, strides=1, n=3)
        x = inverted_residual_block(x, 160, 3, t=6, alpha=self.alpha, strides=2, n=3)
        x = inverted_residual_block(x, 320, 3, t=6, alpha=self.alpha, strides=1, n=1)
        x = Conv_2D_block(x, 1280, 1, 1)

        outputs = self.MLP(x)
        model = tf.keras.Model(inputs, outputs)

        return model

    def MobileNet_v3_Small(self):
        inputs = tf.keras.Input((self.length, self.num_channel))

        x = Conv_2D_block_2(inputs, 16, 3, strides=2, nl='HS')
        x = bottleneck_block_2(x, 16, 3, e=16, s=2, squeeze=True, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=72, s=2, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=88, s=1, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=96, s=2, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 48, 5, e=120, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 48, 5, e=144, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=288, s=2, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = Conv_2D_block_2(x, 576, 1, strides=1, nl='HS')
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
        x = tf.keras.layers.Conv2D(1280, 1, padding='same')(x)
        
        outputs = self.MLP(x)
        model = tf.keras.Model(inputs, outputs)

        return model

    def MobileNet_v3_Large(self):
        inputs = tf.keras.Input((self.length, self.num_channel))

        x = Conv_2D_block_2(inputs, 16, 3, strides=2, nl='HS')
        x = bottleneck_block_2(x, 16, 3, e=16, s=1, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=64, s=2, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 24, 3, e=72, s=1, squeeze=False, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=72, s=2, squeeze=True, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=120, s=1, squeeze=True, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 40, 5, e=120, s=1, squeeze=True, nl='RE', alpha=self.alpha)
        x = bottleneck_block_2(x, 80, 5, e=240, s=2, squeeze=False, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 80, 3, e=200, s=1, squeeze=False, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 80, 3, e=184, s=1, squeeze=False, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 80, 3, e=184, s=1, squeeze=False, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 112, 3, e=480, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 112, 3, e=672, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 160, 5, e=672, s=2, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 160, 5, e=960, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = bottleneck_block_2(x, 160, 5, e=960, s=1, squeeze=True, nl='HS', alpha=self.alpha)
        x = Conv_2D_block_2(x, 960, 1, strides=1, nl='HS')
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
        x = tf.keras.layers.Conv2D(1280, 1, padding='same')(x)

        outputs = self.MLP(x)
        model = tf.keras.Model(inputs, outputs)

        return model
