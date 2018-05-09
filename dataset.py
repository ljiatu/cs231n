import os

from skimage import io
from torch.utils.data import Dataset


class IMDbFacialDataset(Dataset):
    """
    IMDb facial age detection dataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Keep track of how many images are in each subdir.
        # First enumerate all subdirs, and then count the file in subdirs.
        subdirs = os.listdir(self.root_dir)
        self.counts = [0] * len(subdirs)
        for subdir in subdirs:
            self.counts[int(subdir)] = len(os.listdir(os.path.join(self.root_dir, subdir)))
        print(self.counts)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        image = io.imread(self._get_image_name(idx))
        if self.transform:
            image = self.transform(image)

        return image

    def _get_image_name(self, idx):
        if idx >= self.__len__():
            raise IndexError(f'Index {idx} out of bounds!')

        for dir_idx, count in enumerate(self.counts):
            if idx >= count:
                idx -= count
            else:
                return os.listdir(os.path.join(self.root_dir, '{0:0=2d}'.format(dir_idx)))[idx]
