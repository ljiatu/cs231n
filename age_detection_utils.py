import torch
from torch.nn import functional as F


def check_result(scores, y) -> (int, int):
    num_classes = scores.size(1)
    expected_classes = (
        (F.softmax(scores, dim=1) * torch.arange(end=num_classes).cuda()).sum(dim=1).round().type(torch.cuda.LongTensor)
    )
    num_correct = (expected_classes == y.type(torch.cuda.LongTensor)).sum()
    num_samples = scores.size(0)
    return num_correct, num_samples


def get_age_bucket(file_path):
    """
    Extracts the DOB and the year the photo was taken can calculates the age of the person.

    Note that we use age buckets instead of the exact age.

    Args:
        file_path: Full path to the image file.
    Returns:
        Age of the person in the file.
    """
    file_name = file_path.split('/')[-1]
    parts = file_name.split('.')[0].split('_')
    dob = int(parts[-2].split('-')[0])
    photo_token = int(parts[-1])
    return photo_token - dob
