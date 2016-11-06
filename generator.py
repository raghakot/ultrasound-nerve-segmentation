import numpy as np
from keras.preprocessing.image import Iterator


class CustomDataGenerator(Iterator):
    """
    Modified keras ImageDataGenerator that can run on both image and mask.
    Returns data batches in the form img, [has_mask, mask] in order to train multi-output model.
    """

    def __init__(self, X, y, transform_fn,
                 batch_size=32, shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.transform_fn = transform_fn
        super(CustomDataGenerator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock so it can be done in parallel
        batch_img = []
        batch_has_mask = []
        batch_mask = []
        for i, j in enumerate(index_array):
            x_i, y_i = self.transform_fn(self.X[j], self.y[j])
            batch_img.append(x_i)
            batch_has_mask.append(1 if np.count_nonzero(y_i) > 0 else 0)
            batch_mask.append(y_i)

        inputs = np.array(batch_img)
        outputs = {
            'aux_output': np.array(batch_has_mask),
            'main_output': np.array(batch_mask)
        }
        return inputs, outputs
