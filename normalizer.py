import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from add_channel import AddChannel
from dataset import IMDbFacialDataset


def main():
    """
    Calculates the mean and standard deviation of all the images.

    Mean and standard deviation are calculated per RGB channel.
    """
    transform = transforms.Compose([
        AddChannel(),
    ])
    dataset = IMDbFacialDataset('imdb_crop', transform)
    loader = DataLoader(dataset, batch_size=400, num_workers=6)

    running_mean = []
    running_std0 = []
    running_std1 = []
    for i, (x, _) in enumerate(loader, 0):
        # shape (batch_size, 3 or 1, height, width)
        np_array = x.numpy()

        batch_mean = np.mean(np_array, axis=(0, 2, 3))
        batch_std0 = np.std(np_array, axis=(0, 2, 3))
        batch_std1 = np.std(np_array, axis=(0, 2, 3), ddof=1)

        running_mean.append(batch_mean)
        running_std0.append(batch_std0)
        running_std1.append(batch_std1)

        running_mean = np.array(running_mean).mean(axis=0)
        running_std0 = np.array(running_std0).mean(axis=0)
        running_std1 = np.array(running_std1).mean(axis=0)

    print('-' * 50)
    print(running_mean)
    print(running_std0)
    print(running_std1)
    print('-' * 50)


if __name__ == '__main__':
    main()
