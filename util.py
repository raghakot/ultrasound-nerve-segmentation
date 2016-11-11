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


def examine_generator():
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("all")
    from generator import CustomDataGenerator
    from train import transform, filter_mask_presence

    X_val, y_val = filter_mask_presence(X_val, y_val)
    generator = CustomDataGenerator(X_val, y_val, lambda x, y: transform(x, y, augment=True), 32)
    imgs, outs = generator.next()
    for i in range(len(imgs)):
        cv2.imshow("image", imgs[i, 0])
        cv2.imshow("mask", outs['main_output'][i, 0])
        cv2.waitKey(0)


def inspect_set(train=False):
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("all")
    X = X_train if train else X_val
    y = y_train if train else y_val

    from model import build_model
    from train import transform
    from submission import post_process_mask

    model = build_model()
    model.load_weights('./results/net.hdf5')

    for i in range(X.shape[0]):
        img_i, mask_i = transform(X[i], y[i])
        masks, has_mask = model.predict(np.array([img_i]), verbose=1)

        print has_mask[0, 0]

        # print has_masks[0, 0]
        cv2.imshow("Image with mask".format(i), generate_image_with_mask(X[i], y[i]))
        cv2.imshow("pred mask".format(i), post_process_mask(masks[0, 0]))
        cv2.waitKey(0)


if __name__ == '__main__':
    inspect_set(train=False)
