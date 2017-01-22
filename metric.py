import tensorflow as tf
from keras import backend as K

smooth = 1


def dice(y_true, y_pred):
    """
    Average dice across all samples
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)


def dice_strict(y_true, y_pred):
    """
    Average of dice across each image sample
    """
    # Workaround for shape bug.
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)


def dice_loss_strict(y_true, y_pred):
    return -dice_strict(y_true, y_pred)


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def _tf_bce(output, target, from_logits=False):
    """Workaround for keras bug with latest tensorflow"""

    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=target)


def bce(y_true, y_pred):
    # Workaround for shape bug.
    y_true.set_shape(y_pred.get_shape())
    return K.mean(_tf_bce(y_pred, y_true), axis=-1)


# Sanity check loss functions..
if __name__ == "'__main__":
    import numpy as np

    zero = np.zeros(shape=(1, 80, 96))
    non_zero = np.zeros(shape=(1, 80, 96))

    for row in range(40, 60):
        for col in range(40, 60):
            non_zero[0, row, col] = 1.

    y_true = np.array([non_zero])
    y_pred = np.array([non_zero])

    y_true = K.variable(y_true)
    y_pred = K.variable(y_pred)

    print K.eval(dice_loss(y_true, y_pred))
