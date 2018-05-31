import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms

from add_channel import AddChannel
from chalearn_dataset import ChaLearnDataset
from soft_argmax import SoftArgmaxLoss
from trainer import Trainer

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')

    # Load the pretrained RESNET-18 model.
    model = torch.load('models/model.pt')
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
        transforms.Normalize([0.5797703, 0.43427974, 0.38307136], [0.25409877, 0.22383073, 0.21819368]),
    ])
    val_transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5797703, 0.43427974, 0.38307136], [0.25409877, 0.22383073, 0.21819368]),
    ])
    train_dataset = ChaLearnDataset(
        ['ChaLearn/images/train_1', 'ChaLearn/images/train_2'],
        'ChaLearn/gt/train_gt.csv',
        train_transform,
    )
    val_dataset = ChaLearnDataset(
        ['ChaLearn/images/valid'],
        'ChaLearn/gt/valid_gt.csv',
        val_transform,
    )
    test_dataset = ChaLearnDataset(
        ['ChaLearn/images/test_1', 'ChaLearn/images/test_2'],
        'ChaLearn/gt/test_gt.csv',
        train_transform,
    )
    loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.RandomSampler(train_dataset),
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.RandomSampler(val_dataset),
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.RandomSampler(test_dataset),
    )

    return loader_train, loader_val, loader_test


if __name__ == '__main__':
    main()
