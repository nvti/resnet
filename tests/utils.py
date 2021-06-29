import tensorflow as tf


def assert_output_shape(block, input_shape, output_shape):
    input = tf.zeros(input_shape)
    output = block(input)

    assert output.shape == output_shape
