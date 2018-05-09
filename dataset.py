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
        self.subdir_num_images = {
            subdir: len(os.listdir(os.path.join(self.root_dir, subdir)))
            for subdir in subdirs
        }

    def __len__(self):
        return sum(self.subdir_num_images.values())

    def __getitem__(self, idx):
        image = io.imread(self._get_image_name(idx))
        if self.transform:
            image = self.transform(image)

        return image

    def _get_image_name(self, idx):
        if idx >= self.__len__():
            raise IndexError(f'Index {idx} out of bounds!')

        for subdir, count in self.subdir_num_images.items():
            if idx >= count:
                idx -= count
            else:
                return os.listdir(os.path.join(self.root_dir, subdir))[idx]
