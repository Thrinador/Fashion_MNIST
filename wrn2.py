from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


# Depth of the network. Compute N = (n - 4) / 6.
N = 6

# Width of the network.
k = 4

# Number of output classes
nb_classes = 10

# Dropout percent for the network
dropout = 0.3


def batch_layer(partial_model):
    return BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(partial_model)


def activation_layer(partial_model):
    return Activation('relu')(partial_model)


def conv_layer(partial_model, filters, kernel_size, strides=1):
    return Convolution2D(filters, kernel_size, padding='same', strides=strides, kernel_initializer='he_normal',
                         use_bias=False)(partial_model)


def max_pool_layer(partial_model):
    return MaxPooling2D(pool_size=(2, 2))(partial_model)


def average_pool_layer(partial_model):
    return AveragePooling2D(pool_size=(2, 2))(partial_model)


def initial_section(input_layer):
    partial_model = conv_layer(input_layer, 16, (3, 3))
    partial_model = batch_layer(partial_model)
    partial_model = activation_layer(partial_model)
    return partial_model


def make_half_block(partial_model, size):
    partial_model = batch_layer(partial_model)
    partial_model = activation_layer(partial_model)
    partial_model = conv_layer(partial_model, size, (3, 3))
    return partial_model


def split_section(input_model, chunk_size):
    lhs = input_model
    rhs = make_half_block(input_model, chunk_size * k)

    if dropout > 0.0:
        rhs = Dropout(dropout)(rhs)

    rhs = make_half_block(rhs, chunk_size * k)
    return Add()([lhs, rhs])


def expand_conv(input_model, base, strides=(1, 1)):
    lhs = conv_layer(input_model, base * k, (3, 3), strides)
    lhs = batch_layer(lhs)
    lhs = activation_layer(lhs)
    lhs = conv_layer(lhs, base * k, (3, 3))

    rhs = conv_layer(input_model, base * k, (1, 1), strides)

    return Add()([lhs, rhs])


def middle_section(partial_model, chunk_size, strides=(1,1)):
    partial_model = expand_conv(partial_model, chunk_size, strides)
    for _ in range(N - 1):
        partial_model = split_section(partial_model, chunk_size)
    partial_model = batch_layer(partial_model)
    partial_model = activation_layer(partial_model)
    return partial_model


def out_section(partial_model):
    partial_model = AveragePooling2D((7, 7))(partial_model)
    partial_model = Flatten()(partial_model)
    partial_model = Dense(nb_classes, activation='softmax')(partial_model)
    return partial_model


def create_wide_residual_network(input_dim):
    input_layer = Input(shape=input_dim)
    partial_model = initial_section(input_layer)

    partial_model = middle_section(partial_model, 16)
    partial_model = middle_section(partial_model, 32, (2,2))
    partial_model = middle_section(partial_model, 64, (2,2))

    partial_model = out_section(partial_model)
    return Model(input_layer, partial_model)
