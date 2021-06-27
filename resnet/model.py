from .components.layer import ResLayer

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, AvgPool2D, Dense


class ResNet(Model):
    def __init__(self, layers, num_classes=1000, use_bottleneck=False):

        assert len(
            layers) == 4, "layers should be a 4-element list, got {}".format(layers)

        # The model start with a 7x7 convolutions and a MaxPool2D
        self.net = Sequential([
            Conv2D(filters=2,
                   kernel_size=(7, 7),
                   strides=2,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(3, 3), strides=2),
        ])

        # Add 4 layers to the network
        filters = [64, 128, 256, 512]
        for i in range(len(layers)):
            # don't use downsample in the first layer
            use_downsample = (i != 0)
            self.net.add(ResLayer(filter_num=filters[i],
                                  num_block=layers[i],
                                  use_downsample=use_downsample,
                                  use_bottleneck=use_bottleneck))

        # classification network
        self.classifier = Sequential([
            AvgPool2D(pool_size=(1, 1)),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.net(inputs, *args, **kwargs)

        return self.classifier(x, *args, **kwargs)


class ResNet18(ResNet):
    """ResNet18 use BasicBlock and have [2, 2, 2, 2] number of block in each layer
    """

    def __init__(self, num_classes=1000):
        super().__init__(layers=[2, 2, 2, 2],
                         num_classes=num_classes,
                         use_bottleneck=False)


class ResNet34(ResNet):
    """ResNet34 use BasicBlock and have [3, 4, 6, 3] number of block in each layer
    """

    def __init__(self, num_classes=1000):
        super().__init__(layers=[3, 4, 6, 3],
                         num_classes=num_classes,
                         use_bottleneck=False)


class ResNet50(ResNet):
    """ResNet50 use BottleneckBlock and have [3, 4, 6, 3] number of block in each layer
    """

    def __init__(self, num_classes=1000):
        super().__init__(layers=[3, 4, 6, 3],
                         num_classes=num_classes,
                         use_bottleneck=True)


class ResNet101(ResNet):
    """ResNet101 use BottleneckBlock and have [3, 4, 23, 3] number of block in each layer
    """

    def __init__(self, num_classes=1000):
        super().__init__(layers=[3, 4, 23, 3],
                         num_classes=num_classes,
                         use_bottleneck=True)


class ResNet152(ResNet):
    """ResNet152 use BottleneckBlock and have [3, 8, 36, 3] number of block in each layer
    """

    def __init__(self, num_classes=1000):
        super().__init__(layers=[3, 8, 36, 3],
                         num_classes=num_classes,
                         use_bottleneck=True)
