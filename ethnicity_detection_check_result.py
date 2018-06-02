def check_result(scores, y) -> (int, int):
    pred = scores.argmax(1)
    num_correct = (pred == y).sum()
    num_samples = scores.size(0)
    return num_correct, num_samples
