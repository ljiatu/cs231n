import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import transforms

from add_channel import AddChannel
from chalearn_dataset import ChaLearnDataset

MODEL_PATH = 'models/model.pt'
OUTPUT_FILE_NAME = 'ChaLearn/output.txt'
BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    model = torch.load(MODEL_PATH)

    transform = transforms.Compose([
        AddChannel(),
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
                expected_ages = (
                    (F.softmax(scores, dim=1) * torch.arange(end=num_classes).to(device=device))
                    .sum(dim=1).round().type(torch.cuda.LongTensor)
                )
                lines = [f'{file_name},{age}' for file_name, age in zip(file_names, expected_ages)]
                output.writelines(lines)
