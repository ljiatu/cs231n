import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms, models

from age_detection_utils import check_result
from chalearn_training_dataset import ChaLearnTrainingDataset
from constants import NUM_AGE_BUCKETS
from soft_argmax import SoftArgmaxLoss
from trainer import Trainer

BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
MODEL_PATH = 'models/model_imdb_wiki_norm_0001.pt'


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    # Load the pretrained RESNET-18 model.
    model = models.resnet18(pretrained=True)
    model = model.to(device=device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
    model.load_state_dict(torch.load(MODEL_PATH))
    loss_func = SoftArgmaxLoss().to(device=device)
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
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5797703, 0.43427974, 0.38307136], [0.25409877, 0.22383073, 0.21819368]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5797703, 0.43427974, 0.38307136], [0.25409877, 0.22383073, 0.21819368]),
    ])
    train_dataset = ChaLearnTrainingDataset(
        ['ChaLearn/images/train_1', 'ChaLearn/images/train_2'],
        'ChaLearn/gt/train_gt.csv',
        train_transform,
    )
    val_dataset = ChaLearnTrainingDataset(
        ['ChaLearn/images/valid'],
        'ChaLearn/gt/valid_gt.csv',
        val_transform,
    )
    test_dataset = ChaLearnTrainingDataset(
        ['ChaLearn/images/test_1', 'ChaLearn/images/test_2'],
        'ChaLearn/gt/test_gt.csv',
        val_transform,
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
