import numpy as np
import cv2

from data import DataManager


def _grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))


def generate_image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = _grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0
    # Channels = bgr
    img_color[mask_edges, 0] = 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 255

    return img_color


def generate_image_with_masks(img, mask_true, mask_pred):
    # returns a copy of the image with edges of the mask added in red
    img_color = _grays_to_RGB(img)
    mask_edges_true = cv2.Canny(mask_true, 100, 200) > 0
    mask_edges_pred = cv2.Canny(mask_pred, 100, 200) > 0

    # Channels = bgr
    img_color[mask_edges_true, 0] = 0
    img_color[mask_edges_true, 1] = 255
    img_color[mask_edges_true, 2] = 0
    img_color[mask_edges_pred, 2] = 255

    return img_color


def inspect_preds_on_test():
    imgs = DataManager.load_test_data()
    mask = np.load('imgs_mask_test.npy')

    import random
    indices = range(imgs.shape[0])
    random.shuffle(indices)

    for i in indices:
        mask_i = mask[i, 0].astype('float32')
        # See images with mask
        if np.count_nonzero(mask_i) > 0:
            img_i = imgs[i, 0]
            mask_i = cv2.threshold(mask_i, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
            mask_i *= 255
            cv2.imshow("image with mask", generate_image_with_mask(img_i, mask_i))
            cv2.waitKey(0)


def inspect_val():
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("all")

    from model import build_model
    from train import transform

    seg_model = build_model()
    seg_model.load_weights('./results/seg.hdf5')

    for i in range(X_val.shape[0]):
        img_i, mask_i = transform(X_val[i], y_val[i])
        mask_pred = seg_model.predict(np.array([img_i]), verbose=1).astype('float32')[0, 0]

        mask_true = y_val[i]
        # if np.count_nonzero(mask_true) == 0:
        # img = cv2.imread("./data/test/{}.tif".format(i+1), cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("image with mask", generate_image_with_mask(img, mask_true))
        cv2.imshow("pred", mask_pred)
        cv2.imshow("true", mask_true)
        cv2.waitKey(0)

        # img = generate_image_with_mask(X_val[i], mask_true)
        # # print has_mask
        # cv2.imshow("Image with mask".format(i), img)
        # cv2.imshow("pred mask".format(i), mask_pred)
        # cv2.waitKey(0)


if __name__ == '__main__':
    inspect_val()
