from __future__ import print_function

import os
import numpy as np
from augmentation.ImageUtils import load_img, list_images
from sklearn.cross_validation import train_test_split


class DataManager(object):
    DATA_PATH = './input/'

    # Source size = 580 X 420
    IMG_ORIG_ROWS = 420     # Height
    IMG_ORIG_COLS = 580     # Width

    IMG_TARGET_ROWS = 160
    IMG_TARGET_COLS = 160

    @staticmethod
    def create_train_data():
        train_data_path = os.path.join(DataManager.DATA_PATH, 'train')
        images = list_images(train_data_path)
        total = len(images) / 2

        patient_classes = np.ndarray(total, dtype=np.uint8)
        imgs = np.ndarray((total, DataManager.IMG_ORIG_ROWS, DataManager.IMG_ORIG_COLS), dtype=np.uint8)
        imgs_mask = np.ndarray((total, DataManager.IMG_ORIG_ROWS, DataManager.IMG_ORIG_COLS), dtype=np.uint8)

        print('Creating training images...')
        i = 0
        for image_path in images:
            if 'mask' in image_path:
                continue

            image_name = os.path.basename(image_path)
            name = image_name.split('.')[0]
            patient_classes[i] = int(name.split('_')[0])

            image_mask_name = name + '_mask.tif'
            imgs[i] = load_img(os.path.join(train_data_path, image_name), grayscale=True)
            imgs_mask[i] = load_img(os.path.join(train_data_path, image_mask_name), grayscale=True)

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1

        # Build all data set
        print('Saving train samples...')
        mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in imgs_mask]
        DataManager.save_train_val_split(imgs, imgs_mask, "all", stratify=mask_labels)

    @staticmethod
    def create_test_data():
        train_data_path = os.path.join(DataManager.DATA_PATH, 'test')
        images = os.listdir(train_data_path)
        total = len(images)

        imgs = np.ndarray((total, 1, DataManager.IMG_ORIG_ROWS, DataManager.IMG_ORIG_COLS), dtype=np.uint8)
        imgs_id = np.ndarray((total, ), dtype=np.int32)

        print('Creating test images...')
        i = 0
        for image_path in images:
            image_name = os.path.basename(image_path)
            img_id = int(image_name.split('.')[0])
            img = load_img(os.path.join(train_data_path, image_name), grayscale=True)
            img = np.array([img])

            imgs[i] = img
            imgs_id[i] = img_id

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1

        # Build all data set
        print('Saving test samples...')
        imgs = imgs[np.argsort(imgs_id)]
        np.save(os.path.join(DataManager.DATA_PATH, 'imgs_test.npy'), imgs)
        print('Saving to .npy files done.')

    @staticmethod
    def save_train_val_split(X, y, name_prefix, stratify=None, split_ratio=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=stratify, test_size=split_ratio)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_X_train.npy'.format(name_prefix)), X_train)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_X_val.npy'.format(name_prefix)), X_val)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_y_train.npy'.format(name_prefix)), y_train)
        np.save(os.path.join(DataManager.DATA_PATH, '{}_y_val.npy'.format(name_prefix)), y_val)
        print('Saving {} .npy files done.'.format(name_prefix))


    @staticmethod
    def load_train_val_data(name_prefix):
        X_train = np.load(os.path.join(DataManager.DATA_PATH, '{}_X_train.npy'.format(name_prefix)))
        X_val = np.load(os.path.join(DataManager.DATA_PATH, '{}_X_val.npy'.format(name_prefix)))
        y_train = np.load(os.path.join(DataManager.DATA_PATH, '{}_y_train.npy'.format(name_prefix)))
        y_val = np.load(os.path.join(DataManager.DATA_PATH, '{}_y_val.npy'.format(name_prefix)))
        return X_train, X_val, y_train, y_val

    @staticmethod
    def load_test_data():
        return np.load(os.path.join(DataManager.DATA_PATH, 'imgs_test.npy'))


if __name__ == '__main__':
    dm = DataManager()
    dm.create_train_data()
    dm.create_test_data()
