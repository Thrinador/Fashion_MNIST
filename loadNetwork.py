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
test = pd.read_csv(r"C:\Users\benji\Desktop\Test\test.csv").values

# Reshape and normalize test data
testX = test[:, 1:].reshape(test.shape[0], 1, 28, 28).astype('float32')
X_test = testX / 255.0

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

ynew = loaded_model.predict_proba(X_test)
t = compress_labels(ynew)
for index in range(0, len(t)):
    print("%s,%s" % ((index + 60001), t[index]))

# show the inputs and predicted outputs
#for i in range(len(X_test)):
#    print("%s, %s" % (X_test[i], ynew[i]))
