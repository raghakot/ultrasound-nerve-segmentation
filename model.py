from keras.layers import (
    Input,
    merge,
    Flatten,
    Dense
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
)
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.optimizers import Adam
from metric import dice_loss, dice
from data import DataManager


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=(1, 1), init="he_normal",
                             border_mode="same")(input)
        norm = BatchNormalization()(conv)
        return ELU()(norm)
    return f


def build_model(optimizer=None):
    if optimizer is None:
        optimizer = Adam(lr=1e-3)

    inputs = Input((1, DataManager.IMG_TARGET_ROWS, DataManager.IMG_TARGET_COLS), name='main_input')
    conv1 = _conv_bn_relu(32, 5, 5)(inputs)
    conv1 = _conv_bn_relu(32, 5, 5)(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = _conv_bn_relu(64, 5, 5)(pool1)
    conv2 = _conv_bn_relu(64, 5, 5)(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = _conv_bn_relu(128, 5, 5)(pool2)
    conv3 = _conv_bn_relu(128, 5, 5)(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = _conv_bn_relu(256, 5, 5)(pool3)
    conv4 = _conv_bn_relu(256, 5, 5)(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = _conv_bn_relu(512, 5, 5)(pool4)
    conv5 = _conv_bn_relu(512, 5, 5)(conv5)

    # Head for scoring nerve presence.
    pre = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    up6 = merge([UpSampling2D()(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = _conv_bn_relu(256, 5, 5)(up6)
    conv6 = _conv_bn_relu(256, 5, 5)(conv6)

    up7 = merge([UpSampling2D()(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = _conv_bn_relu(128, 5, 5)(up7)
    conv7 = _conv_bn_relu(128, 5, 5)(conv7)

    up8 = merge([UpSampling2D()(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = _conv_bn_relu(64, 5, 5)(up8)
    conv8 = _conv_bn_relu(64, 5, 5)(conv8)

    up9 = merge([UpSampling2D()(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = _conv_bn_relu(32, 5, 5)(up9)
    conv9 = _conv_bn_relu(32, 5, 5)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init="he_normal", name='main_output')(conv9)

    model = Model(input=inputs, output=[conv10, aux_out])
    model.compile(optimizer=optimizer,
                  loss={'main_output': dice_loss, 'aux_output': 'binary_crossentropy'},
                  metrics={'main_output': dice, 'aux_output': 'acc'},
                  loss_weights={'main_output': 1., 'aux_output': 0.5})
    return model


if __name__ == '__main__':
    model = build_model()
