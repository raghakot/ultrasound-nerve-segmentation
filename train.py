from __future__ import print_function

from datetime import datetime

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from generator import CustomDataGenerator
from augmentation.ImageAugmenter import ImageAugmenter
from data import DataManager
from model import build_model

run_id = str(datetime.now())
tb = TensorBoard(log_dir='./logs/{}'.format(run_id), histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4, min_lr=1e-6)
augmenter = ImageAugmenter(DataManager.IMG_ORIG_COLS, DataManager.IMG_ORIG_ROWS,
                           hflip=False, vflip=False,
                           rotation_deg=5,
                           translation_x_px=10,
                           translation_y_px=10)


def filter_mask_presence(imgs, masks, presence=True):
    """
    Extracts samples by mask presence
    """
    has_mask = np.where([np.count_nonzero(mask) > 0 for mask in masks])
    if not presence:
        has_mask = not has_mask
    return imgs[has_mask], masks[has_mask]


def preprocess(img, denoise=False):
    """
    Preprocess step after image augmentation, and before feeding into conv net.
    """
    if denoise:
        img = cv2.fastNlMeansDenoising(img, h=7)
    img = cv2.resize(img, (DataManager.IMG_TARGET_COLS, DataManager.IMG_TARGET_ROWS))
    return img


def transform(img, mask, augment=True):
    """
    Transforms an (img, mask) pair with same augmentation params
    """
    if augment:
        img, mask = augmenter.augment_batch(np.array([img, mask]), same_transform=True)
    img = preprocess(img)
    mask = preprocess(mask).astype('float32') / 255.
    return np.array([img]), np.array([mask])


def train(resume=False):
    print('Loading data...')
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("cleaned")
    # X_train, y_train = filter_mask_presence(X_train, y_train)
    # X_val, y_val = filter_mask_presence(X_train, y_train)

    print('Creating and compiling model...')
    model = build_model()
    if resume:
        model.load_weights('./results/net.hdf5')
    model_checkpoint = ModelCheckpoint('./results/net.hdf5', monitor='val_loss', save_best_only=True)

    print('Training on model')
    model.summary()
    batch_size = 64
    nb_epoch = 200

    train_generator = CustomDataGenerator(X_train, y_train, transform, batch_size)

    # Use fixed samples instead to visualize histograms. There is currently a bug that prevents it
    # when a val generator is used.
    # Not aug val samples to keep the eval consistent.
    val_generator = CustomDataGenerator(X_val, y_val, lambda x, y: transform(x, y, augment=False), batch_size)

    model.fit_generator(train_generator, validation_data=val_generator, nb_val_samples=X_val.shape[0],
                        samples_per_epoch=X_train.shape[0], nb_epoch=nb_epoch, verbose=2,
                        callbacks=[model_checkpoint, reduce_lr, tb], max_q_size=1000)


if __name__ == '__main__':
    train(resume=False)
