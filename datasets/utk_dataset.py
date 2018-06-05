import os

from skimage import io
from torch.utils.data import Dataset

# Divide into 5 buckets - asian, caucasian, indian, black and others.
NUM_ETHNICITY_BUCKETS = 5


class UTKDataset(Dataset):
    """
    UTK Face dataset.
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        self.file_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = f'race/UTKFace/{self.file_names[idx]}'
        image = io.imread(file_path)
        if self.transform:
            image = self.transform(image)

        race = int(file_path.split('_')[2])

        return image, race
