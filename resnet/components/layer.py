from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from .block import BuildingBlock


class ResLayer(Layer):
    def __init__(self, filter_num, num_block, use_downsample=False, use_bottleneck=False):

        self.net = Sequential()
        if use_downsample:
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=2,
                                       use_downsample=use_downsample,
                                       use_bottleneck=use_bottleneck))
        else:
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=1,
                                       use_downsample=use_downsample,
                                       use_bottleneck=use_bottleneck))

        for _ in range(num_block - 1):
            self.net.add(BuildingBlock(filter_num=filter_num,
                                       stride=1,
                                       use_downsample=False,
                                       use_bottleneck=use_bottleneck))

    def call(self, inputs, *args, **kwargs):
        return self.net(inputs, *args, **kwargs)
