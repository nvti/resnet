from resnet import *

from argparse import ArgumentParser
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--model', default='resnet34', type=str,
                        help='Type of ResNet model, valid option: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--lr', default=0.01,
                        type=float, help='Learning rate')

    args = parser.parse_args()

    if args.model == 'resnet18':
        model = ResNet18(args.classes)
    elif args.model == 'resnet34':
        model = ResNet34(args.classes)
    elif args.model == 'resnet50':
        model = ResNet50(args.classes)
    elif args.model == 'resnet101':
        model = ResNet101(args.classes)
    elif args.model == 'resnet152':
        model = ResNet152(args.classes)
    else:
        raise "Invalid model name. Valid option: resnet18, resnet34, resnet50, resnet101, resnet152"

    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
