import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from constants import ETHNICITIES, NUM_AGE_BUCKETS
from datasets.utk_dataset import NUM_ETHNICITY_BUCKETS


class AgethNet(torch.nn.Module):
    """
    A CNN predicting age based on ethnicity.
    """
    def __init__(self, ethnicity_model_path, device):
        super(AgethNet, self).__init__()

        self.device = device

        self.ethnicity_model = models.resnet50(pretrained=True)
        self.ethnicity_model = self.ethnicity_model.to(device=self.device)
        num_ftrs = self.ethnicity_model.fc.in_features
        self.ethnicity_model.fc = nn.Linear(num_ftrs, NUM_ETHNICITY_BUCKETS).to(device=self.device)
        self.ethnicity_model.load_state_dict(torch.load(ethnicity_model_path))
        self.ethnicity_model.eval()

        for ethnicity in ETHNICITIES:
            model = models.resnet18(pretrained=True)
            model = model.to(device=self.device)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_AGE_BUCKETS).to(device=self.device)
            self.add_module(ethnicity, model)

    def forward(self, x):
        with torch.no_grad():
            ethnicity_scores = self.ethnicity_model(x)
            ethnicity_probabilities = F.softmax(ethnicity_scores, dim=1)

        # The 0th position of the ethnicity array must correspond to the same ethnicity in the predicted age array.
        predicted_ages = []
        for ethnicity in ETHNICITIES:
            age_scores = self._modules[ethnicity](x)
            predicted_age = (
                (F.softmax(age_scores, dim=1) * torch.arange(end=NUM_AGE_BUCKETS, device=self.device, requires_grad=True)).sum(dim=1)
            )
            predicted_ages.append(predicted_age)

        ages_tensor = torch.stack(predicted_ages, dim=1)

        return (ethnicity_probabilities * ages_tensor).sum(dim=1).round()
