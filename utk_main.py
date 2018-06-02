import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from add_channel import AddChannel
from ethnicity_detection_check_result import check_result
from trainer import Trainer
from utk_dataset import NUM_ETHNICITY_BUCKETS, UTKDataset

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
IMAGE_DIR = 'race/UTKFace'


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
    model.fc = nn.Linear(num_ftrs, NUM_ETHNICITY_BUCKETS).cuda()
    loss_func = CrossEntropyLoss().cuda()
    # dtype depends on the loss function.
    dtype = torch.cuda.LongTensor
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loader_train, loader_val, loader_test = _split_data()
    model_trainer = Trainer(
        model, loss_func, dtype, optimizer, device,
        loader_train, loader_val, loader_test, check_result,
        num_epochs=5, print_every=500
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
        transforms.Normalize([0.59702533, 0.4573939, 0.3917105], [0.25691032, 0.22929442, 0.22493552]),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.59702533, 0.4573939, 0.3917105], [0.25691032, 0.22929442, 0.22493552]),
    ])
    train_dataset = UTKDataset(IMAGE_DIR, train_transform)
    val_dataset = UTKDataset(IMAGE_DIR, val_transform)
    test_dataset = UTKDataset(IMAGE_DIR, val_transform)
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
