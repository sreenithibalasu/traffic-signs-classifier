import keras
import tensorflow as tf
import numpy as np
from training_model import training_model
from data_preprocessing import preprocess, check_consistency
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import datetime
import random
import json
import pickle
import os
import sys
import argparse

if __name__ == '__main__':

    # Command line args
    """
    train_file - pickle filename with training data including ground truth
    batch_size - number of samples to be taken during one iteration/epoch
    epochs - number of training iterations the model has to run for
    val_file - pickle filename with validation data including ground truth

    Usage notes:
    python3 run_training.py --train_file train.p  --val_file val.p
                            --batch_size 50 --epochs 10
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    # parser.add_argument('--steps_per_epoch', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    # parser.add_argument('--test_file', type=str, required=False)

    args = parser.parse_args()


    f = open("./configs.json", 'r')
    configs = json.load(f)

    data_path = configs['data_path']
    checkpoints_path = configs['checkpoints_path']
    log_path_train = configs['log_dir_train']

    # Open data files
    with open(os.path.join(data_path, args.train_file), 'rb') as f:
        train_set = pickle.load(f)

    with open(os.path.join(data_path, args.val_file), 'rb') as f:
        validation_set = pickle.load(f)

    # Load data
    x_train, y_train = train_set['features'], train_set['labels']
    x_val, y_val = validation_set['features'], validation_set['labels']

    print('Training data shape: ', x_train.shape)
    print('Validation data shape: ', x_val.shape)

    check_consistency(x_train, y_train, "train")
    check_consistency(x_val, y_val, "validation")

    #Preprocess data
    x_train = np.array(list(map(preprocess, x_train)))
    x_val= np.array(list(map(preprocess, x_val)))

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_val = x_val.reshape(x_val.shape[0], 32, 32, 1)

    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)


    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)
    datagen.fit(x_train)

    # Build the model
    model = training_model(configs['num_classes'])
    print(model.summary())

    checkpoints_dir = os.path.dirname(checkpoints_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    steps_per_epoch = len(x_train) // args.batch_size
    val_steps = len(x_val) // args.batch_size

    log_dir_train = os.path.join(log_path_train, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_train, histogram_freq=1)

    # Train the model
    model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size), steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs, validation_data=(x_val, y_val), shuffle=1, callbacks=[cp_callback, tensorboard_callback])

    print('MODEL FINISHED TRAINING. CHECKPOINTS SAVED AT {}'.format(checkpoints_dir))
