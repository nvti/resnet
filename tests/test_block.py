from resnet.components.block import *
from tensorflow.keras import Sequential


def test_basic_block():
    model = Sequential([BasicBlock(filter_num=64, stride=1)])
    model.build(input_shape=(None, 28, 28, 1))

    # output channel = filter_num
    assert model.output_shape == (None, 28, 28, 64)

    model = Sequential([BasicBlock(filter_num=64, stride=2)])
    model.build(input_shape=(None, 28, 28, 1))
    # output size /= 2
    assert model.output_shape == (None, 14, 14, 64)


def test_bottleneck_block():
    model = Sequential([BottleNeckBlock(filter_num=64, stride=1)])
    model.build(input_shape=(None, 28, 28, 1))
    # output channel = filter_num * 4
    assert model.output_shape == (None, 28, 28, 256)

    model = Sequential([BottleNeckBlock(filter_num=64, stride=2)])
    model.build(input_shape=(None, 28, 28, 1))
    # output size /= 2
    assert model.output_shape == (None, 14, 14, 256)


def test_building_block():
    model = Sequential([BuildingBlock(filter_num=64, stride=2,
                                      use_downsample=True, use_bottleneck=False)])
    model.build(input_shape=(None, 28, 28, 1))
    assert model.output_shape == (None, 14, 14, 64)

    model = Sequential([BuildingBlock(filter_num=64, stride=2,
                                      use_downsample=True, use_bottleneck=True)])
    model.build(input_shape=(None, 28, 28, 1))
    assert model.output_shape == (None, 14, 14, 256)
