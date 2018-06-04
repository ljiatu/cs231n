import os
import shutil

import torch
from skimage import io
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models

from add_channel import AddChannel

ETHNICITIES = ['caucasian', 'black', 'asian', 'indian', 'others']


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device {device}')
    
    ethnicity_model = models.resnet50(pretrained=True)
    ethnicity_model = ethnicity_model.to(device=device)
    num_ftrs = ethnicity_model.fc.in_features
    ethnicity_model.fc = nn.Linear(num_ftrs, 5).to(device=device)
    ethnicity_model.load_state_dict(torch.load('models/utk_model_resnet_50.pt'))
    ethnicity_model.eval()

    transform = transforms.Compose([
        AddChannel(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.59702533, 0.4573939, 0.3917105], [0.25691032, 0.22929442, 0.22493552]),
    ])

    # Iterate over each image and determine the ethnicity.
    # If the confidence is high, then directly put in the corresponding directory.
    # Otherwise, calls for human intervention.
    precessed = 0
    low_probability_images = []
    #     for subdir in os.listdir('imdb_wiki'):
    #         print(f'Processing directory {subdir}')
    #         for file_name in os.listdir(f'imdb_wiki/{subdir}'):
    #             file_path = f'imdb_wiki/{subdir}/{file_name}'
    #             image = transform(io.imread(file_path)).unsqueeze(0).to(device=device)
    #             with torch.no_grad():
    #                 scores = ethnicity_model(image)
    #                 probabilities = F.softmax(scores, dim=1)
    #                 probability, ethnicity = probabilities.max(dim=1)
    #                 if probability < 0.6:
    #                     low_probability_images.append(f'{file_path}\n')
    #                 else:
    #                     shutil.copy2(file_path, f'imdb_wiki_ethnicity/{ETHNICITIES[ethnicity]}/')

    #                 precessed += 1

    #         print(f'Processed {precessed} files\n')

    #         with open('uncertain.txt', 'w') as f:
    #             f.writelines(low_probability_images)
    # Try increasing threshould to 0.7.
    still_uncertain = []
    with open('uncertain.txt', 'r') as f:
        file_paths = f.readlines()
        for file_path in file_paths:
            file_path = file_path.strip()
            image = transform(io.imread(file_path)).unsqueeze(0).to(device=device)
            with torch.no_grad():
                scores = ethnicity_model(image)
                probabilities = F.softmax(scores, dim=1)
                probability, ethnicity = probabilities.max(dim=1)
                if probability < 0.5:
                    still_uncertain.append(f'{file_path},{probability[0]}\n')
                else:
                    shutil.copy2(file_path, f'imdb_wiki_ethnicity/{ETHNICITIES[ethnicity]}/')

    with open('still_uncertain.txt', 'w') as f:
        f.writelines(still_uncertain)

if __name__ == '__main__':
    main()
