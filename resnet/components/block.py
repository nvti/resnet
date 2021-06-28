from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU
from tensorflow.keras import Sequential


class BasicBlock(Layer):
    """`BasicBlock` use stack of two 3x3 convolutions layers
    """

    def __init__(self, filter_num, stride=1):
        """Creates a `BasicBlock` layer instance.

        Args:
        filter_num: the number of filters in the convolution
        stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
        """

        super(BasicBlock, self).__init__()

        self.expansion = 1
        self.net = Sequential([
            Conv2D(filters=filter_num,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=filter_num * self.expansion,
                   kernel_size=(3, 3),
                   strides=1,
                   padding='same'),
            BatchNormalization(),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)


class BottleNeckBlock(Layer):
    """`BottleNeckBlock` use stack of 3 layers: 1x1, 3x3 and 1x1 convolutions
    """

    def __init__(self, filter_num, stride=1):
        """Creates a `BottleNeckBlock` layer instance.

        Args:
        filter_num: the number of filters in the convolution
        stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
        """
        super(BottleNeckBlock, self).__init__()

        self.expansion = 4
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
            Conv2D(filters=filter_num * self.expansion,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same'),
            BatchNormalization(),
        ])

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)


class BuildingBlock(Layer):
    """A complete `BuildingBlock` of ResNet
    """

    def __init__(self, filter_num, stride=1, use_downsample=False, use_bottleneck=False):
        """Creates a `BuildingBlock` layer instance.

        Args:
        filter_num: the number of filters in the convolution
        stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
        use_downsample: type of shortcut connection: Identity or Projection shortcut
        use_bottleneck: type of block: basic or bottleneck
        """

        super(BuildingBlock, self).__init__()

        if use_bottleneck:
            self.net = BottleNeckBlock(filter_num, stride)
        else:
            self.net = BasicBlock(filter_num, stride)

        if use_downsample:
            # create downsample network: a 1x1 convolutions
            self.downsample = Sequential([
                Conv2D(filters=filter_num * self.net.expansion,
                       kernel_size=(1, 1),
                       strides=stride),
                BatchNormalization()
            ])
        else:
            self.downsample = None

        self.relu = ReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.net(inputs, *args, **kwargs)

        # use downsample if need
        if self.downsample != None:
            inputs = self.downsample(inputs, *args, **kwargs)

        # add the input with output of the network
        return self.relu(inputs + x, *args, **kwargs)
