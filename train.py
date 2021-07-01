from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from argparse import ArgumentParser
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='resnet34', type=str,
                        help='Type of ResNet model, valid option: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--classes', default=10,
                        type=int, help='Number of classes')
    parser.add_argument('--lr', default=0.01,
                        type=float, help='Learning rate')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of training epoch')
    parser.add_argument('--image-size', default=224,
                        type=int, help='Size of input image')
    parser.add_argument('--image-channels', default=3,
                        type=int, help='Number channel of input image')
    parser.add_argument('--train-folder', default='', type=str,
                        help='Where training data is located')
    parser.add_argument('--valid-folder', default='', type=str,
                        help='Where validation data is located')
    parser.add_argument('--model-folder', default='.output/',
                        type=str, help='Folder to save trained model')

    args = parser.parse_args()

    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        train_ds = image_dataset_from_directory(
            args.train_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
        val_ds = image_dataset_from_directory(
            args.valid_folder,
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            batch_size=args.batch_size,
        )
    else:
        print("Data folder is not set. Use Fashion MNIST dataset")

        args.image_size = 28
        args.image_channels = 1
        args.classes = 10

        (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
        x_train = (x_train.reshape(-1, args.image_size, args.image_size,
                                   args.image_channels) / 255).astype(np.float32)
        x_val = (x_val.reshape(-1, args.image_size, args.image_size,
                               args.image_channels) / 255).astype(np.float32)

        # create dataset
        train_ds = Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(args.batch_size)

        val_ds = Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.batch(args.batch_size)

    # create model
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
        raise 'Invalid model name. Valid option: resnet18, resnet34, resnet50, resnet101, resnet152'

    model.build(input_shape=(None, args.image_size,
                             args.image_size, args.image_channels))
    model.summary()

    optimizer = Adam(learning_rate=args.lr)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer, loss=loss,
                  metrics=['accuracy'])

    # Traning
    model.fit(train_ds,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=val_ds)

    # Save model
    model.save(args.model_folder)
