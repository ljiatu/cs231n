import bisect
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
        # First enumerate all subdirs, and then count the number of files in subdirs.
        subdirs = os.listdir(self.root_dir)
        counts = [0] * len(subdirs)
        for subdir in subdirs:
            counts[int(subdir)] = len(os.listdir(os.path.join(self.root_dir, subdir)))
        # Roll-up the counts so that searching for an image at a given index takes log time.
        # For example, if counts = [1, 2, 3, 4], then the rolled-up counts will be [1, 3, 6, 10]
        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]
        self.counts = counts

    def __len__(self):
        return self.counts[-1]

    def __getitem__(self, idx):
        image = io.imread(self._get_image_name(idx))
        if self.transform:
            image = self.transform(image)

        return image

    def _get_image_name(self, idx):
        if idx >= self.__len__():
            raise IndexError(f'Index {idx} out of bounds!')

        subdir_idx = bisect.bisect(self.counts, idx)
        subdir_path = os.path.join(self.root_dir, '{0:0=2d}'.format(subdir_idx))
        # Calculate the image index in the subdir.
        image_idx = idx if subdir_idx == 0 else idx - self.counts[subdir_idx - 1]
        name = os.listdir(subdir_path)[image_idx]
        return os.path.join(subdir_path, name)
