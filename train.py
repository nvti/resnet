from .resnet import *

from argparse import ArgumentParser
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--model', default='resnet34', type=str,
                        dest='Type of ResNet model, valid option: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--classes', default=1000,
                        type=int, dest='Number of class')
    parser.add_argument('--lr', default=0.01,
                        type=float, dest='Learning rate')
    parser.add_argument('--momentum', default=0.9,
                        type=float, dest='Learning momentum')

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

    optimizer = SGD(learning_rate=args.lr, momentum=args.momentum)
    loss_fn = CategoricalCrossentropy()
