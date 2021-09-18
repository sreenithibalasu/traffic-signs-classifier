import keras
import tensorflow as tf
import numpy as np
import pandas as pd
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
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':

    """
    System Args

    --test_file - path to the test pickle file containing multiple images
    --image_url - a URL to an image that needs to be tested. Can be multiple URLs


    Usage notes:
    python3 test_model.py --test_file ./test.p --image_url https://...

    Example:
    python3 test_model.py --image_url https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg


    """

    mpl.use('tkagg')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=False)
    parser.add_argument('--image_url', nargs='+', required=False)

    args = parser.parse_args()

    print(args.image_url)

    f = open("./configs.json", 'r')
    configs = json.load(f)

    ckpt_path = configs['checkpoints_path']

    df = pd.read_csv(configs['csv_mapping'])
    test_model = training_model(configs['num_classes'])
    test_model.load_weights(ckpt_path)

    # Check if test data is pickle file or url
    if args.test_file is not None:

        with open(os.path.join(data_path, args.test_file), 'rb') as f:
            test_set = pickle.load(f)
            x_test, y_test = test_set['features'], test_set['labels']
            print('Test data shape: ', x_test.shape)
            check_consistency(x_test, y_test)

            x_test = np.array(list(map(preprocess, x_test)))
            x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
            y_test = to_categorical(y_test, 43)

            score = test_model.evaluate(x_test, y_test, verbose=0)

            print('Test Score: ', score[0])
            print('Test Accuracy:', score[1])

    if args.image_url is not None:
        urls = args.image_url
        for url in urls:
          r = requests.get(url, stream=True)

          img_original = Image.open(r.raw)
          img = np.asarray(img_original)
          img = cv2.resize(img, (32, 32))

          img_processed = preprocess(img)

          img = img_processed.reshape(1, 32, 32, 1)

          predicted_class = test_model.predict_classes(img)
          sign = df[df['ClassId']==int(predicted_class)]['SignName']

          print("predicted class: ", str(predicted_class))
          print("corresponding sign: ", sign.values)

          plt.imshow(img_original)
          plt.axis('off')

          plt.title("corresponding sign: " + str(sign.values))
          plt.waitforbuttonpress()
