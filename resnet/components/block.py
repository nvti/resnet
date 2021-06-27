import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras import Sequential


class BasicBlock(Layer):
    def __init__(self, filter_num, stride=1):
        self.net = Sequential([
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=1,
                   padding='same'),
            BatchNormalization(),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)


class BottleNeckBlock(Layer):
    def __init__(self, filter_num, stride=1):
        self.net = Sequential([
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

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)


class BuildingBlock(Layer):
    def __init__(self, filter_num, stride=1, use_downsample=False, use_bottleneck=False):
        if use_bottleneck:
            self.net = BottleNeckBlock(filter_num, stride)
            self.expansion = 4
        else:
            self.net = BasicBlock(filter_num, stride)
            self.expansion = 1

        if use_downsample:
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
            ReLU()
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.net(inputs, *args, **kwargs)

        if self.downsample != None:
            inputs = self.downsample(inputs, *args, **kwargs)

        return self.add([inputs, x], *args, **kwargs)
