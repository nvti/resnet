from resnet.components.layer import *
from tensorflow.keras import Sequential


def test_layer_basic():
    model = Sequential([ResLayer(filter_num=64, num_block=3,
                                 use_downsample=False, use_bottleneck=False)])
    model.build(input_shape=(None, 28, 28, 1))

    # output channel = filter_num
    assert model.output_shape == (None, 28, 28, 64)


def test_layer_basic_downsample():
    model = Sequential([ResLayer(filter_num=64, num_block=3,
                                 use_downsample=True, use_bottleneck=False)])
    model.build(input_shape=(None, 28, 28, 1))

    # output size /= 2
    assert model.output_shape == (None, 14, 14, 64)


def test_layer_bottleneck():
    model = Sequential([ResLayer(filter_num=64, num_block=3,
                                 use_downsample=False, use_bottleneck=True)])
    model.build(input_shape=(None, 28, 28, 1))

    # output channel = filter_num * 4
    assert model.output_shape == (None, 28, 28, 256)


def test_layer_bottleneck_downsample():
    model = Sequential([ResLayer(filter_num=64, num_block=3,
                                 use_downsample=True, use_bottleneck=True)])
    model.build(input_shape=(None, 28, 28, 1))

    # output size /= 2
    assert model.output_shape == (None, 14, 14, 256)
