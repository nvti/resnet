from resnet.components.layer import ResLayer
from tests.utils import assert_output_shape


def test_layer_basic():
    # test basic layer without down sampling
    assert_output_shape(block=ResLayer(filter_num=64, num_block=3,
                                       use_downsample=False,
                                       use_bottleneck=False),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 28, 28, 64))  # output channel = filter_num


def test_layer_basic_downsample():
    # test basic layer without down sampling
    assert_output_shape(block=ResLayer(filter_num=64, num_block=3,
                                       use_downsample=True,
                                       use_bottleneck=False),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 64))  # output size /= 2


def test_layer_bottleneck():
    # test bottleneck layer without down sampling
    assert_output_shape(block=ResLayer(filter_num=64, num_block=3,
                                       use_downsample=False,
                                       use_bottleneck=True),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 28, 28, 256))  # output channel = filter_num * 4


def test_layer_bottleneck_downsample():
    # test bottleneck layer without down sampling
    assert_output_shape(block=ResLayer(filter_num=64, num_block=3,
                                       use_downsample=True,
                                       use_bottleneck=True),
                        input_shape=(1, 28, 28, 1),
                        output_shape=(1, 14, 14, 256))  # output size /= 2
