import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from dataset import IMDbFacialDataset
from trainer import Trainer

BATCH_SIZE = 500
NUM_AGE_BUCKETS = 100


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Use a pretrained RESNET-50 model. Freeze all the layers except the last FC layer.
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device=device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_AGE_BUCKETS)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    loader_train, loader_val, loader_test = _split_data()
    model_trainer = Trainer(model, loss_func, optimizer, device, loader_train, loader_val)
    model_trainer.train()


def _split_data():
    transform = transforms.Compose([
        AddChannel(),
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

    return loader_train, loader_val, loader_test


if __name__ == '__main__':
    main()
