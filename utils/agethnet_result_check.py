import torch


def check_result(predicted_ages, y) -> (int, int):
    num_correct = (predicted_ages.type(torch.cuda.LongTensor) == y.type(torch.cuda.LongTensor)).sum()
    num_samples = predicted_ages.size(0)
    return num_correct, num_samples
