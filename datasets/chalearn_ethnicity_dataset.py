import csv
import os

from skimage import io
from torch.utils.data import Dataset


class ChaLearnEthnicityDataset(Dataset):
    """
    ChaLearn dataset with images split into different directories based on ethnicities.
    """

    def __init__(self, image_dir: str, label_file_path: str, transform=None):
        """
        Args:
            image_dir: Directory with all the images of a specific ethnicity.
            transform: Optional transform to be applied on a sample.
        """
        self.transform = transform

        self.num_images = len(os.listdir(image_dir))
        self.image_file_paths = [f'{image_dir}/{file_name}' for file_name in os.listdir(image_dir)]

        with open(label_file_path) as label_file:
            label_reader = csv.reader(label_file)
            # Skip the header row.
            next(label_reader)
            self.labels = {row[0]: row[1] for row in label_reader}

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        file_path = self.image_file_paths[idx]
        image = io.imread(file_path)
        if self.transform:
            image = self.transform(image)

        file_name = file_path.split('/')[-1]
        return image, int(round(float(self.labels[file_name])))
