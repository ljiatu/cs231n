import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import IMDbFacialDataset

BATCH_SIZE = 500


def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = IMDbFacialDataset('imdb_crop', transform)
    # Do a rough 8:1:1 split between training set, validation set and test set.
    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    loader_train = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler.SubsetRandomSampler(range(num_train))
    )
    loader_val = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val))
    )
    loader_test = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler.SubsetRandomSampler(range(num_train + num_val, len(dataset)))
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




if __name__ == '__main__':
    main()
