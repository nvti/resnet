import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras import Sequential


class BasicBlock(Layer):
    def __init__(self, filter_num, stride=1):
        self.block = Sequential([
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding='same'),
            BatchNormalization(),
        ])

    def call(self, inputs):
        return self.block(inputs)


class BottleNeckBlock(Layer):
    def __init__(self, filter_num, stride=1, is_downsample=False):
        self.block = Sequential([
            Conv2D(filters=filter_num,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_num * 4,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same'),
            BatchNormalization(),
        ])

    def call(self, inputs):
        return self.block(inputs)


class BuildBlock(Layer):
    def __init__(self, filter_num, stride=1, is_downsample=False, is_bottleneck=False):
        if is_bottleneck:
            self.block = BottleNeckBlock(filter_num, stride, is_downsample)
            self.expansion = 4
        else:
            self.block = BasicBlock(filter_num, stride, is_downsample)
            self.expansion = 1

        if is_downsample:
            self.downsample = Sequential([
                Conv2D(filters=filter_num * self.expansion,
                       kernel_size=(1, 1),
                       strides=stride),
                BatchNormalization()
            ])
        else:
            self.downsample = None

        self.add = Sequential([
            Add(),
            ReLU(),
        ])

    def call(self, inputs):
        x = self.block(inputs)

        if self.downsample != None:
            inputs = self.downsample(inputs)

        return self.add([inputs, x])
