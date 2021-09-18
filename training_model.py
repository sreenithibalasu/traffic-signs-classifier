import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def training_model(num_classes):
    model = Sequential()


    model.add(Conv2D(64, kernel_size=(5,5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(5,5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    #lower learning rate gives better accuracy for complex datasets
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])
    return model
