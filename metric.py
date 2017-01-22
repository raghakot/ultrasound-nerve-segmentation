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
