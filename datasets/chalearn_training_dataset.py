from datasets.chalearn_dataset import ChaLearnDataset


class ChaLearnTrainingDataset(ChaLearnDataset):
    """
    ChaLearn Look At People age prediction dataset for training.

    This dataset returns the actual age instead of the file name.
    """
    def __getitem__(self, idx):
        image, file_name = super().__getitem__(idx)
        return image, int(round(float(self.labels[file_name])))
