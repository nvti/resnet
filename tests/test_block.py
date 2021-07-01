from resnet.components.block import *
from tests.utils import assert_output_shape
import tensorflow as tf


def test_basic_block():
    # test basic block with stride=1 (same image size)
    assert_output_shape(block=BasicBlock(filter_num=64, stride=1),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 28, 28, 64))  # output channel = filter_num

    # test basic block with stride=2 (down sampling)
    assert_output_shape(block=BasicBlock(filter_num=64, stride=2),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 64))


def test_bottleneck_block():
    # test bottleneck block with stride=1 (same image size)
    assert_output_shape(block=BottleNeckBlock(filter_num=64, stride=1),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 28, 28, 256))  # output channel = filter_num * 4

    # test bottleneck block with stride=2 (down sampling)
    assert_output_shape(block=BottleNeckBlock(filter_num=64, stride=2),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 256))


def test_building_block():

    # test basic block with down sampling (use projection shortcut)
    assert_output_shape(block=BuildingBlock(filter_num=64, stride=2,
                                            use_downsample=True,
                                            use_bottleneck=False),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 64))

    # test bottleneck block with down sampling (use projection shortcut)
    assert_output_shape(block=BuildingBlock(filter_num=64, stride=2,
                                            use_downsample=True,
                                            use_bottleneck=True),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 256))
