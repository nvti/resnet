from .layers.residual import ResBlock

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, MaxPool2D


class ResNet(Model):
    def __init__(self, layers, num_classes=100, is_bottleneck=False):
        super(ResNet, self).__init__()

        if len(layers) != 4:
            raise ValueError(
                "layers should be a 4-element list, got {}".format(layers))

        self.net = Sequential([
            Conv2D(filters=2,
                   kernel_size=(7, 7),
                   strides=2,
                   padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=(3, 3), strides=2),
        ])
