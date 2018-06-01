import csv
import os
from typing import List

from skimage import io
from torch.utils.data import Dataset

# Divide ages into 101 buckets, which represent ages [0, 100] inclusive.
NUM_AGE_BUCKETS = 101


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

        self.length = sum([len(os.listdir(image_dir)) for image_dir in image_dirs])

        self.image_file_paths = []
        for image_dir in image_dirs:
            file_paths = os.listdir(image_dir)
            self.image_file_paths.extend(f'{image_dir}/{file_path}' for file_path in file_paths)

        # Load labels into a dict for fast lookup.
        with open(label_file_path) as label_file:
            label_reader = csv.reader(label_file)
            # Skip the header row.
            next(label_reader)
            self.labels = {row[0]: row[1] for row in label_reader}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_path = self.image_file_paths[idx]
        image = io.imread(file_path)
        if self.transform:
            image = self.transform(image)

        file_name = file_path.split('/')[-1]
        return image, file_name  # int(round(float(self.labels[file_name])))
