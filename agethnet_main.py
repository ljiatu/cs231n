import sys
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import models, transforms

from constants import NUM_AGE_BUCKETS, ETHNICITIES
from datasets.chalearn_training_dataset import ChaLearnDataset

ETHNICITY_MODEL_PATH = 'models/utk_model_resnet_50.pt'
BATCH_SIZE = 400
DATA_LOADER_NUM_WORKERS = 10
MODES = ['expected', 'max']


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in MODES:
        raise ValueError(f"Mode must be specified, and be one of {MODES}")

    mode = sys.argv[1]
    output_file_path = f'ChaLearn/{sys.argv[2]}'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    # First, load ethnicity classification model.
    ethnicity_model = models.resnet50(pretrained=True)
    ethnicity_model = ethnicity_model.to(device=device)
    num_ftrs = ethnicity_model.fc.in_features
    ethnicity_model.fc = torch.nn.Linear(num_ftrs, len(ETHNICITIES)).to(device=device)
    ethnicity_model.load_state_dict(torch.load(ETHNICITY_MODEL_PATH))
    ethnicity_model.eval()

    # Then, load ethnicity-specific age detection model.
    age_models = []
    for ethnicity in ETHNICITIES:
        age_model = models.resnet18(pretrained=True)
        age_model = age_model.to(device=device)
        num_ftrs = age_model.fc.in_features
        age_model.fc = torch.nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=device)
        age_model.load_state_dict(torch.load(f'models/agethnet-{ethnicity}.pt'))
        age_model.eval()
        age_models.append(age_model)

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
        with open(output_file_path, 'w') as output:
            for x, file_names in loader:
                x = x.to(device=device)
                ethnicity_scores = ethnicity_model(x)
                ethnicity_probabilities = F.softmax(ethnicity_scores, dim=1)

                age_scores = torch.zeros(x.size()[0], len(ETHNICITIES), NUM_AGE_BUCKETS).to(device=device)
                for i in range(len(ETHNICITIES)):
                    age_scores[:, i, :] = age_models[i](x)

                per_ethnicity_ages = (
                    (F.softmax(age_scores, dim=2) * torch.arange(end=NUM_AGE_BUCKETS, device=device)).sum(dim=2)
                )

                if mode == 'expected':
                    # Calculates an expected age using all ethnicity probabilities.
                    predicted_ages = (ethnicity_probabilities * per_ethnicity_ages).sum(dim=1).round()
                else:
                    # Use the age from the most likely ethnicity.
                    ethnicity_idx = ethnicity_probabilities.argmax(dim=1)
                    predicted_ages = per_ethnicity_ages.gather(1, ethnicity_idx.view(-1, 1))[0].round()

                lines = [f'{file_name},{age}\n' for file_name, age in zip(file_names, predicted_ages)]
                output.writelines(lines)


if __name__ == '__main__':
    main()
