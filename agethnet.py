import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F


from imdb_wiki_dataset import NUM_AGE_BUCKETS

ETHNICITIES = ['caucasian', 'black', 'asian', 'indian', 'others']


class AgethNet(torch.nn.Module):
    """
    A CNN predicting age based on ethnicity.
    """
    def __init__(self, ethnicity_model_path, device, dtype):
        super(AgethNet, self).__init__()

        self.ethnicity_model = torch.load(ethnicity_model_path)
        self.ethnicity_model.eval()
        self.device = device
        self.dtype = dtype

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
        predicted_ages = torch.zeros(x.shape[0], len(ETHNICITIES)).type(self.dtype)
        for i, ethnicity in enumerate(ETHNICITIES):
            age_scores = self._modules[ethnicity](x)
            predicted_age = (
                (F.softmax(age_scores, dim=1) * torch.arange(end=NUM_AGE_BUCKETS).to(device=self.device)).sum(dim=1)
            )
            predicted_ages[:, i] = predicted_age

        return (ethnicity_probabilities * predicted_ages).sum(dim=1)
