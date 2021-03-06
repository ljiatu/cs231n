import csv
import os
from typing import List

from skimage import io
from torch.utils.data import Dataset


class ChaLearnDataset(Dataset):
    """
    ChaLearn Look At People age prediction dataset.
    """

    def __init__(self, image_dirs: List[str], label_file_path: str, transform=None):
        """
        Args:
            image_dirs: Directory with all the images.
            label_file_path: Path to the ground truth label file.
            transform: Optional transform to be applied on a sample.
        """
        self.transform = transform

        self.num_images = sum([len(os.listdir(image_dir)) for image_dir in image_dirs])

        self.image_file_paths = []
        for image_dir in image_dirs:
            file_names = os.listdir(image_dir)
            self.image_file_paths.extend(f'{image_dir}/{file_name}' for file_name in file_names)

        # Load labels into a dict for fast lookup.
        with open(label_file_path) as label_file:
            label_reader = csv.reader(label_file)
            # Skip the header row.
            next(label_reader)
            self.labels = {row[0]: row[1] for row in label_reader}

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        file_path = self.image_file_paths[idx]
        image = io.imread(file_path)
        if self.transform:
            image = self.transform(image)

        file_name = file_path.split('/')[-1]
        return image, file_name
