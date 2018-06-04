import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from agethnet_result_check import check_result
from constants import NUM_AGE_BUCKETS, ETHNICITIES
from imdb_wiki_ethnicity_dataset import IMDbWikiEthnicityDataset
from soft_argmax import SoftArgmaxLoss
from trainer import Trainer

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
IMAGE_DIR = 'imdb_wiki_ethnicity'


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    for ethnicity in ETHNICITIES:
        model_path = f'models/agethnet_{ethnicity}'
        # Use a pretrained RESNET-18 model.
        model = models.resnet18(pretrained=True)
        model = model.to(device=device)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).cuda()
        loss_func = SoftArgmaxLoss().cuda()
        # dtype depends on the loss function.
        dtype = torch.cuda.FloatTensor
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        loader_train, loader_val, loader_test = _split_data(ethnicity)
        model_trainer = Trainer(
            model, loss_func, dtype, optimizer, device,
            loader_train, loader_val, loader_test, check_result,
            model_path, num_epochs=10, print_every=400
        )
        model_trainer.train()
        model_trainer.test()


def _split_data(ethnicity):
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
    train_dataset = IMDbWikiEthnicityDataset(f'{IMAGE_DIR}/{ethnicity}', train_transform)
    val_dataset = IMDbWikiEthnicityDataset(f'{IMAGE_DIR}/{ethnicity}', val_transform)
    test_dataset = IMDbWikiEthnicityDataset(f'{IMAGE_DIR}/{ethnicity}', val_transform)
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
