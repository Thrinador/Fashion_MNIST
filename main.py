from keras.layers.convolutional import Convolution2D
from sklearn import preprocessing
from keras import backend as K
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import wrn2
import numpy as np
import sys


# Some variable to tweak to mess with the performance of the network.
number_of_epochs = int(sys.argv[2])
batch_size = 200
iterations = 2


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def run_network(p_train, t_train):
    model = wrn2.create_wide_residual_network((28, 28, 1))

    imgn_train = ImageDataGenerator(rotation_range=10,
                                    zoom_range=0.10,
                                    width_shift_range=5./32,
                                    height_shift_range=5./32,
                                    horizontal_flip=True)
    imgn_train.fit(p_train)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(
        imgn_train.flow(*shuffle(p_train, t_train), batch_size=batch_size),
        steps_per_epoch=p_train.shape[0] // (4 * batch_size),
        epochs=number_of_epochs)
    return model


def run_many_networks(p_train, t_train, p_test, t_test):
    max_accuracy = 0
    mean_accuracy = 0
    max_model = None
    for i in range(1, iterations + 1):
        print("Train number:", i)
        model = run_network(p_train, t_train)
        score, acc = model.evaluate(p_test, t_test,
                                    batch_size=batch_size)

        print("Accuracy:" + str(acc))

        mean_accuracy += acc
        if acc > max_accuracy:
            max_accuracy = acc
            max_model = model
        print()
        print()

    mean_accuracy /= iterations
    return [max_model, max_accuracy, mean_accuracy]


def print_network_summary(model, max_acc, mean_acc, output_file):
    with open(output_file + "_summary.txt", "w") as summary_file:
        summary_file.write("Max Accuracy was:  " + str(max_acc) + "\n")
        summary_file.write("Mean Accuracy was: " + str(mean_acc) + "\n")
        summary_file.write("\n")
        summary_file.write("Number of Epochs: " + str(number_of_epochs) + "\n")
        summary_file.write("Size of Batch: " + str(batch_size) + "\n")
        summary_file.write("Number of Iterations: " + str(iterations) + "\n")
        summary_file.write("\n")
        model.summary(print_fn=summary_file.write)


def save_network(model, output_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open(output_file + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(output_file + ".h5")


def get_input_data():
    # Read training and test data files
    train = pd.read_csv(r"C:\Users\benji\OneDrive\Desktop\FashionMNIST\Test\train.csv").values
    test = pd.read_csv(r"C:\Users\benji\OneDrive\Desktop\FashionMNIST\Test\test.csv").values
    testX = test[:, 1:].reshape(test.shape[0], 28, 28, 1).astype('float32')
    # Reshape and normalize training data
    train_x = train[:, 1:].reshape(train.shape[0], 28, 28, 1).astype('float32')

    full_dataset = np.concatenate((train_x, testX));

    total_mean = np.mean(full_dataset)
    total_std = np.std(full_dataset)
    train_x -= total_mean
    p_train = train_x / total_std
    t_train = train[:, 0]

    # Do some pre-processing on the t_training data.
    lb = preprocessing.LabelBinarizer()
    t_train = lb.fit_transform(t_train)

    return [p_train, t_train]


def main(output_file):
    [p_train, t_train] = get_input_data()

    # p_test = p_train[:3000]
    # p_train = p_train[3000:]
    #
    # t_test = t_train[:3000]
    # t_train = t_train[3000:]

    p_test = p_train
    t_test = t_train

    # Trains the network the iteration times each time with the number of epochs and batch sizes given
    [model, max_acc, mean_acc] = run_many_networks(p_train, t_train, p_test, t_test)

    # Writes a brief summary of the model to a summary.txt file
    print_network_summary(model, max_acc, mean_acc, output_file)

    # Saves the model to a .json and .h5 file.
    save_network(model, output_file)


if __name__ == "__main__":
    main(sys.argv[1])
