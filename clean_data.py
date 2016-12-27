import os
import shutil
import numpy as np

import skimage.util
import skimage.io
import scipy.spatial.distance as spdist

from collections import defaultdict
from data import DataManager


def _compute_img_hist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)


def _are_inconsistent(mask1, mask2):
    has_mask1 = np.count_nonzero(mask1) > 0
    has_mask2 = np.count_nonzero(mask2) > 0
    return has_mask1 != has_mask2


def _filter_inconsistent(imgs, masks):
    hists = np.array(map(_compute_img_hist, imgs))
    dists = spdist.squareform(spdist.pdist(hists, metric='cosine'))

    # + eye because image will be similar to itself. We dont want to include those.
    close_pairs = dists + np.eye(dists.shape[0]) < 0.008
    close_ij = np.transpose(np.nonzero(close_pairs))

    # Find inconsistent masks among duplicates
    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in close_ij:
        if _are_inconsistent(masks[i], masks[j]):
            valids[i] = valids[j] = False

    return np.array(imgs)[valids], np.array(masks)[valids]


def create_cleaned():
    # Clean up old data.
    out_dir = './input/train_cleaned'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Group by patient id.
    patient_classes, imgs, imgs_mask = DataManager.read_train_images()
    pid_data_dict = defaultdict(list)
    for i, pid in enumerate(patient_classes):
        pid_data_dict[pid].append((imgs[i], imgs_mask[i]))

    imgs_cleaned = []
    imgs_masks_cleaned = []
    for pid in pid_data_dict:
        imgs, masks = zip(*pid_data_dict[pid])
        filtered_imgs, filtered_masks = _filter_inconsistent(imgs, masks)
        print("Discarded {} from patient {}".format(len(imgs) - len(filtered_imgs), pid))
        imgs_cleaned.extend(filtered_imgs)
        imgs_masks_cleaned.extend(filtered_masks)

    imgs = np.array(imgs_cleaned)
    imgs_mask = np.array(imgs_masks_cleaned)
    print("Creating cleaned train dataset: {} items".format(len(imgs)))
    mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in imgs_mask]
    DataManager.save_train_val_split(imgs, imgs_mask, "cleaned", stratify=mask_labels)


if __name__ == "__main__":
    create_cleaned()
