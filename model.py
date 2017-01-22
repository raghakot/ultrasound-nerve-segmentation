from keras.layers import (
    Input,
    merge,
    Flatten,
    BatchNormalization,
    Dropout
)
from keras.layers.convolutional import (
    Convolution2D,
    UpSampling2D
)
from keras.layers.advanced_activations import ELU
from keras.models import Model

from keras.optimizers import Adam
from metric import dice_loss, dice, bce
from data import DataManager


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=subsample, init="he_normal",
                             border_mode="same")(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


def build_model(optimizer=None):
    if optimizer is None:
        optimizer = Adam(lr=1e-4)

    inputs = Input((1, DataManager.IMG_TARGET_ROWS, DataManager.IMG_TARGET_COLS), name='main_input')
    conv1 = _conv_bn_relu(32, 7, 7)(inputs)
    conv1 = _conv_bn_relu(32, 3, 3)(conv1)
    pool1 = _conv_bn_relu(32, 2, 2, subsample=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = _conv_bn_relu(64, 3, 3)(drop1)
    conv2 = _conv_bn_relu(64, 3, 3)(conv2)
    pool2 = _conv_bn_relu(64, 2, 2, subsample=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = _conv_bn_relu(128, 3, 3)(drop2)
    conv3 = _conv_bn_relu(128, 3, 3)(conv3)
    pool3 = _conv_bn_relu(128, 2, 2, subsample=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = _conv_bn_relu(256, 3, 3)(drop3)
    conv4 = _conv_bn_relu(256, 3, 3)(conv4)
    pool4 = _conv_bn_relu(256, 2, 2, subsample=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = _conv_bn_relu(512, 3, 3)(drop4)
    conv5 = _conv_bn_relu(512, 3, 3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Using conv to mimic fully connected layer.
    aux = Convolution2D(nb_filter=1, nb_row=drop5._keras_shape[2], nb_col=drop5._keras_shape[3],
                        subsample=(1, 1), init="he_normal", activation='sigmoid')(drop5)
    aux = Flatten(name='aux_output')(aux)

    up6 = merge([UpSampling2D()(drop5), conv4], mode='concat', concat_axis=1)
    conv6 = _conv_bn_relu(256, 3, 3)(up6)
    conv6 = _conv_bn_relu(256, 3, 3)(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = merge([UpSampling2D()(drop6), conv3], mode='concat', concat_axis=1)
    conv7 = _conv_bn_relu(128, 3, 3)(up7)
    conv7 = _conv_bn_relu(128, 3, 3)(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = merge([UpSampling2D()(drop7), conv2], mode='concat', concat_axis=1)
    conv8 = _conv_bn_relu(64, 3, 3)(up8)
    conv8 = _conv_bn_relu(64, 3, 3)(conv8)
    drop8 = Dropout(0.5)(conv8)

    up9 = merge([UpSampling2D()(drop8), conv1], mode='concat', concat_axis=1)
    conv9 = _conv_bn_relu(32, 3, 3)(up9)
    conv9 = _conv_bn_relu(32, 3, 3)(conv9)
    drop9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init="he_normal", name='main_output')(drop9)

    model = Model(input=inputs, output=[conv10, aux])
    model.compile(optimizer=optimizer,
                  loss={'main_output': dice_loss, 'aux_output': bce},
                  metrics={'main_output': dice, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1, 'aux_output': 0.5})

    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
