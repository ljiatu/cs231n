import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from dataset import IMDbFacialDataset, NUM_AGE_BUCKETS
from trainer import Trainer

BATCH_SIZE = 500
DATA_LOADER_NUM_WORKERS = 5


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    # Use a pretrained RESNET-18 model.
    model = models.resnet18(pretrained=True)
    model = model.to(device=device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_AGE_BUCKETS).cuda()
    loss_func = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loader_train, loader_val, loader_test = _split_data()
    model_trainer = Trainer(
        model, loss_func, optimizer, device,
        loader_train, loader_val, loader_test,
        num_epochs=10, print_every=100
    )
    model_trainer.train()
    model_trainer.test()


def _split_data():
    train_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = IMDbFacialDataset('imdb_crop', train_transform)
    val_dataset = IMDbFacialDataset('imdb_crop', val_transform)
    test_dataset = IMDbFacialDataset('imdb_crop', val_transform)
    # Do a rough 8:1:1 split between training set, validation set and test set.
    num_train = int(len(train_dataset) * 0.8)
    num_val = int(len(val_dataset) * 0.1)
    loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.SubsetRandomSampler(range(num_train))
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val))
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.SubsetRandomSampler(range(num_train + num_val, len(test_dataset)))
    )

    return loader_train, loader_val, loader_test


if __name__ == '__main__':
    main()
