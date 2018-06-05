import os

from skimage import io
from torch.utils.data import Dataset

from utils.age_detection_utils import get_age_bucket


class IMDbWikiEthnicityDataset(Dataset):
    """
    IMDb-Wiki dataset split into different ethnicities.
    """

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: Directory with all the images of a specific ethnicity.
            transform: Optional transform to be applied on a sample.
        """
        self.transform = transform

        self.num_images = len(os.listdir(image_dir))
        self.image_file_paths = [f'{image_dir}/{file_name}' for file_name in os.listdir(image_dir)]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        file_path = self.image_file_paths[idx]
        image = io.imread(file_path)
        if self.transform:
            image = self.transform(image)

        return image, get_age_bucket(file_path)
