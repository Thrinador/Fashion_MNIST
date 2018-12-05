import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from keras import backend as K
from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D
from sklearn import preprocessing
from keras import backend as K
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda, Flatten
from keras.layers import InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import os
import sys


def compress_labels(labels):
    t = []
    for label in labels:
        max_location = 0
        max_val = 0

        for index in range(0, len(label)):
            if label[index] > max_val:
                max_val = label[index]
                max_location = index

        t.append(max_location)
    return t


# Read test data file
train = pd.read_csv(r"C:\Users\benji\OneDrive\Desktop\FashionMNIST\Test\train.csv").values
test = pd.read_csv(r"C:\Users\benji\OneDrive\Desktop\FashionMNIST\Test\test.csv").values
testX = test[:, 1:].reshape(test.shape[0], 28, 28, 1).astype('float32')
# Reshape and normalize training data
train_x = train[:, 1:].reshape(train.shape[0], 28, 28, 1).astype('float32')

full_dataset = np.concatenate((train_x, testX));

total_mean = np.mean(full_dataset)
total_std = np.std(full_dataset)

# Reshape and normalize test data
testX -= np.mean(total_mean)
X_test = testX / total_std

# load json and create model
json_file = open(sys.argv[1] + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(sys.argv[1] + ".h5")
print("Id,label")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

ynew = loaded_model.predict(X_test)
t = compress_labels(ynew)
for index in range(0, len(t)):
    print("%s,%s" % ((index + 60001), t[index]))
