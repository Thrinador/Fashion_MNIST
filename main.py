import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn import preprocessing
from keras import backend as K
import sys

def run_network(x_train, y_train, number_of_epochs, batch_size):
    # Sets the shape type for the network
    K.set_image_dim_ordering('th')

    # Sets up the network structure
    # almost all of the items here need to be tweaked and tested.
    model = Sequential()
    model.add(Convolution2D(56, (5, 5), input_shape=(1, 28, 28), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    # optimizer: Adam is a new and popular choice that slightly outperforms other optimizers while still running fast
    # loss: categorical_crossentropy is supposed to be the best approach when we have a categorical output.
    # metrics: I want to see the accuray of the model as the network learns
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the network
    # epochs: free to adjust, aiming for the highest number without over learning.
    # batch_size: free to adjust, lower the number the slower the learning, needs to be balenced with number of epochs
    history = model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=batch_size)
    model.summary()

    return [history, model]


def main():
    # Read training and test data files
    train = pd.read_csv(r"C:\Users\benji\Desktop\Test\train.csv").values

    # Reshape and normalize training data
    train_x = train[:, 1:].reshape(train.shape[0], 1, 28, 28).astype('float32')
    x_train = train_x / 255.0
    y_train = train[:, 0]

    lb = preprocessing.LabelBinarizer()

    y_train = lb.fit_transform(y_train)

    number_of_epochs = 20
    batch_size = 200

    max_accuracy = 0
    mean_accuracy = 0
    max_model = None
    for i in range(1, 2):
        print("train number:", i)
        [history, model] = run_network(x_train, y_train, number_of_epochs, batch_size)
        acc = history.history['acc'][number_of_epochs-1]
        mean_accuracy += acc
        if acc > max_accuracy:
            max_accuracy = acc
            max_model = model
        print()
        print()

    mean_accuracy /= 1
    print("Max Accuracy was:  ", max_accuracy)
    print("Mean Accuracy was: ", mean_accuracy)
    # serialize model to JSON
    model_json = max_model.to_json()
    with open(sys.argv[1] + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    max_model.save_weights(sys.argv[1] + ".h5")

main()