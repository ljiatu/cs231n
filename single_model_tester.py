import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import models, transforms

from datasets.chalearn_training_dataset import ChaLearnDataset
from constants import NUM_AGE_BUCKETS

MODEL_PATH = 'models/model_imdb_wiki_norm_0001_epoch20.pt'
OUTPUT_FILE_NAME = 'ChaLearn/output_0001_epoch20.csv'
BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    model = models.resnet18(pretrained=True)
    model = model.to(device=device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5797703, 0.43427974, 0.38307136], [0.25409877, 0.22383073, 0.21819368]),
    ])
    dataset = ChaLearnDataset(
        ['ChaLearn/images/test_1', 'ChaLearn/images/test_2'],
        'ChaLearn/gt/test_gt.csv',
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.RandomSampler(dataset),
    )

    # Test and write the results to a file.
    with torch.no_grad():
        with open(OUTPUT_FILE_NAME, 'w') as output:
            for x, file_names in loader:
                x = x.to(device=device)
                scores = model(x)
                num_classes = scores.size(1)
                predicted_ages = (
                    (F.softmax(scores, dim=1) * torch.arange(end=num_classes).to(device=device)).sum(dim=1)
                )
                lines = [f'{file_name},{age}\n' for file_name, age in zip(file_names, predicted_ages)]
                output.writelines(lines)


if __name__ == '__main__':
    main()
