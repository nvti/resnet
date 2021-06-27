from .block import BuildingBlock

from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential


class ResLayer(Layer):
    """A complete `ResLayer` of ResNet
    """

    def __init__(self, filter_num, num_block, use_downsample=False, use_bottleneck=False):
        """Creates a `ResLayer` layer instance.

        Args:
        filter_num: the number of filters in the convolution
        num_block: number of `BuildingBlock` in a layer
        use_downsample: type of shortcut connection: Identity or Projection shortcut
        use_bottleneck: type of block: basic or bottleneck
        """

        self.net = Sequential()
        if use_downsample:
            # The first block use downsample (stride=2)
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=2,
                                       use_downsample=True,
                                       use_bottleneck=use_bottleneck))
        else:
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=1,
                                       use_downsample=False,
                                       use_bottleneck=use_bottleneck))

        # Add other block to network
        for _ in range(1, num_block):
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=1,
                                       use_downsample=False,
                                       use_bottleneck=use_bottleneck))

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)
