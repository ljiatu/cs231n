import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from dataset import IMDbFacialDataset, NUM_AGE_BUCKETS
from soft_argmax import SoftArgmaxLoss
from trainer import Trainer

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 4


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
    loss_func = SoftArgmaxLoss().cuda()
    # dtype depends on the loss function.
    dtype = torch.cuda.FloatTensor
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loader_train, loader_val, loader_test = _split_data()
    model_trainer = Trainer(
        model, loss_func, dtype, optimizer, device,
        loader_train, loader_val, loader_test,
        num_epochs=10, print_every=200
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
        transforms.Normalize([0.44118115, 0.3723839, 0.3325232], [0.2887202, 0.26569104, 0.26193744]),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.44118115, 0.3723839, 0.3325232], [0.2887202, 0.26569104, 0.26193744]),
    ])
    train_dataset = IMDbFacialDataset('imdb_crop', train_transform)
    val_dataset = IMDbFacialDataset('imdb_crop', val_transform)
    test_dataset = IMDbFacialDataset('imdb_crop', val_transform)
    # Do a rough 98:1:1 split between training set, validation set and test set.
    num_train = int(len(train_dataset) * 0.98)
    num_val = int(len(val_dataset) * 0.01)
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
