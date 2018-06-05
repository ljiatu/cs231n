import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import models, transforms

from constants import ETHNICITIES, NUM_AGE_BUCKETS
from datasets.imdb_wiki_ethnicity_dataset import IMDbWikiEthnicityDataset
from loss_funcs.soft_argmax import SoftArgmaxLoss
from trainer import Trainer
from utils.add_channel import AddChannel
from utils.age_detection_utils import check_result

BATCH_SIZE = 350
DATA_LOADER_NUM_WORKERS = 10
IMAGE_DIR = 'imdb_wiki_ethnicity'
EPOCHS = [10, 5, 2, 3, 1]
NORMS = [
    [[0.58159447, 0.43522802, 0.36891466], [0.24821207, 0.21232615, 0.20570053]],
    [[0.45464125, 0.324485, 0.2616199], [0.23923942, 0.19709928, 0.18780835]],
    [[0.59417844, 0.45388743, 0.38434005], [0.2409343, 0.21290545, 0.20881842]],
    [[0.5519065, 0.40091512, 0.3221176], [0.24537557, 0.20903918, 0.19956224]],
    [[0.5589975, 0.40766674, 0.33462912], [0.26535666, 0.21328007, 0.19708937]],
]


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    for ethnicity, num_epochs, norm in zip(['black', 'asian', 'indian', 'others'], EPOCHS, NORMS):
        model_path = f'models/agethnet-{ethnicity}.pt'
        # Use a pretrained RESNET-18 model.
        model = models.resnet18(pretrained=True)
        model = model.to(device=device)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
        model.load_state_dict(torch.load('models/resnet18-andy.pt'))
        loss_func = SoftArgmaxLoss().to(device=device)
        # dtype depends on the loss function.
        dtype = torch.cuda.FloatTensor
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        loader_train, loader_val, loader_test = _split_data(ethnicity, norm)
        model_trainer = Trainer(
            model, loss_func, dtype, optimizer, device,
            loader_train, loader_val, loader_test, check_result,
            model_path, num_epochs=num_epochs, print_every=400
        )
        model_trainer.train()
        model_trainer.test()


def _split_data(ethnicity, norm):
    train_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(norm[0], norm[1]),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm[0], norm[1]),
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
