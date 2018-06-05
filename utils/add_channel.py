import numpy as np


class AddChannel:
    """
    Some of the input images do not have a channel dimension.
    Augment these images to have a channel dimension.
    """
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image as an numpy.ndarray.

        Returns:
            img: image with at least three dimensions: channel, height, width.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError(f'Input pic must be a NumPy ndarray, not {type(img)}')
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
