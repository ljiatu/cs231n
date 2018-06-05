import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from constants import ETHNICITIES, NUM_AGE_BUCKETS
from datasets.chalearn_ethnicity_dataset import ChaLearnEthnicityDataset
from loss_funcs.soft_argmax import SoftArgmaxLoss
from trainer import Trainer
from utils.age_detection_utils import check_result

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
NORMS = [
    [[0.5857966, 0.43951988, 0.38925144], [0.25121605, 0.22208285, 0.21685465]],
    [[0.46575668, 0.33568966, 0.28021362], [0.25198266, 0.21828137, 0.20614661]],
    [[0.5584561, 0.4057047, 0.3450245], [0.25622535, 0.22345945, 0.2153755]],
    [[0.5277182, 0.39202887, 0.33102724], [0.252444, 0.21967761, 0.21011677]],
    [[0.53532773, 0.40135944, 0.33226442], [0.26975623, 0.22523403, 0.20693415]],
]


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    for ethnicity, norm in zip(ETHNICITIES, NORMS):
        model_path = f'models/agethnet-{ethnicity}.pt'
        # Use a pretrained RESNET-18 model.
        model = models.resnet18(pretrained=True)
        model = model.to(device=device)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
        model.load_state_dict(torch.load(model_path))
        loss_func = SoftArgmaxLoss().to(device=device)
        # dtype depends on the loss function.
        dtype = torch.cuda.FloatTensor
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        loader_train, loader_val, loader_test = _split_data(ethnicity, norm)
        model_trainer = Trainer(
            model, loss_func, dtype, optimizer, device,
            loader_train, loader_val, loader_test, check_result,
            model_path, num_epochs=1, print_every=100
        )
        model_trainer.train()
        model_trainer.test()


def _split_data(ethnicity, norm):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(norm[0], norm[1]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm[0], norm[1]),
    ])
    train_dataset = ChaLearnEthnicityDataset(
        f'ChaLearn/ethnicity/{ethnicity}',
        'ChaLearn/ethnicity/gt.csv',
        train_transform,
    )
    val_dataset = ChaLearnEthnicityDataset(
        f'ChaLearn/ethnicity/{ethnicity}',
        'ChaLearn/ethnicity/gt.csv',
        val_transform,
    )
    test_dataset = ChaLearnEthnicityDataset(
        f'ChaLearn/ethnicity/{ethnicity}',
        'ChaLearn/ethnicity/gt.csv',
        val_transform,
    )
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
