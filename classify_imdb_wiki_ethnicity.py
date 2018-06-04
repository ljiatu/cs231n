import os
import shutil

import torch
from skimage import io
from torch.nn import functional as F
from torchvision.transforms import transforms

from add_channel import AddChannel

ETHNICITIES = ['caucasian', 'black', 'asian', 'indian', 'others']


def main():
    ethnicity_model = torch.load('models/utk_model.pt')
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
    low_probability_images = []
    for subdir in os.listdir('imdb_wiki'):
        for file_name in os.listdir(subdir):
            file_path = f'imdb_wiki/{subdir}/{file_name}'
            image = transform(io.imread(file_path))
            scores = ethnicity_model(image)
            probabilities = F.softmax(scores)
            probability, ethnicity = probabilities.max()
            if probability < 0.5:
                low_probability_images.append(file_path)
            else:
                shutil.copy2(file_path, f'imdb_wiki_ethnicity/{ETHNICITIES[ethnicity]}/')

    print('\n'.join(low_probability_images))
