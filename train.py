from __future__ import print_function

from datetime import datetime

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import SGD

from generator import CustomDataGenerator
from augmentation.ImageAugmenter import ImageAugmenter
from data import DataManager
from model import build_model

run_id = str(datetime.now())
tb = TensorBoard(log_dir='./logs/{}'.format(run_id), histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=0.001)
augmenter = ImageAugmenter(DataManager.IMG_ORIG_COLS, DataManager.IMG_ORIG_ROWS,
                           hflip=True, vflip=True,
                           rotation_deg=10,
                           translation_x_px=20,
                           translation_y_px=20)


def preprocess(img, denoise=False):
    """
    Preprocess step after image augmentation
    """
    # crop to 400 * 400
    img = img[0:400, 180:580]
    if denoise:
        img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.resize(img, (DataManager.IMG_TARGET_COLS, DataManager.IMG_TARGET_ROWS))
    return img


def transform(img, mask):
    """
    Transforms an (img, mask) pair with same augmentation params
    """
    img, mask = augmenter.augment_batch(np.array([img, mask]), same_transform=True)
    img = preprocess(img)
    mask = preprocess(mask)
    return np.array([img]), np.array([mask])


def train():
    print('Loading data...')
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("all")

    print('Creating and compiling model...')
    opt = SGD(lr=0.1, nesterov=True, momentum=0.9)
    model = build_model(opt)
    model_checkpoint = ModelCheckpoint('./results/net.hdf5', monitor='val_loss', save_best_only=True)

    print('Fitting model...')
    batch_size = 32
    nb_epoch = 100

    train_generator = CustomDataGenerator(X_train, y_train, transform, batch_size)
    val_generator = CustomDataGenerator(X_val, y_val, transform, batch_size)
    model.fit_generator(train_generator, validation_data=val_generator, nb_val_samples=X_val.shape[0] * 2,
                        samples_per_epoch=X_train.shape[0], nb_epoch=nb_epoch, verbose=1,
                        callbacks=[model_checkpoint, reduce_lr, tb], max_q_size=10000)


if __name__ == '__main__':
    train()
