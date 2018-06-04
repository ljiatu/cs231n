import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from age_detection_utils import check_result
from imdb_wiki_dataset import IMDbWikiDataset, NUM_AGE_BUCKETS
from soft_argmax import SoftArgmaxLoss
from trainer import Trainer

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
IMAGE_DIR = 'imdb_wiki'
MODEL_PATH = 'models/model.pt'


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
    model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
    loss_func = SoftArgmaxLoss().cuda()
    # dtype depends on the loss function.
    dtype = torch.cuda.FloatTensor
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loader_train, loader_val, loader_test = _split_data()
    model_trainer = Trainer(
        model, loss_func, dtype, optimizer, device,
        loader_train, loader_val, loader_test, check_result,
        MODEL_PATH, num_epochs=5, print_every=100
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
        transforms.Normalize([0.57089275, 0.4255322, 0.35874116], [0.24959293, 0.21301098, 0.20608185]),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.57089275, 0.4255322, 0.35874116], [0.24959293, 0.21301098, 0.20608185]),
    ])
    train_dataset = IMDbWikiDataset(IMAGE_DIR, train_transform)
    val_dataset = IMDbWikiDataset(IMAGE_DIR, val_transform)
    test_dataset = IMDbWikiDataset(IMAGE_DIR, val_transform)
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
