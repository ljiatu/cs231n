import csv
import math


# ------------------------------------------------------
# Main evaluation functions
# ------------------------------------------------------
import sys


def load_labels(fpath):
    labels = {}
    with open(fpath) as fp:
        rd = csv.reader(fp)
        next(rd, None)
        for row in rd:
            labels[row[0]] = {
                'mean': float(row[1]),
                'stdv': float(row[2])
            }
    return labels


def load_predictions(fpath):
    with open(fpath) as fp:
        rd = csv.reader(fp)
        return {row[0]: float(row[1]) for row in rd}


def evaluate_predictions(gt, preds):
    # Calculate non-normalized score
    score, n_scored = (0, 0)
    for k, v in preds.items():
        if k in gt:
            gt[k]['stdv'] = 0.162 if gt[k]['stdv'] < 0.162 else gt[k]['stdv']
            score = score + (1 - math.exp(-(v - gt[k]['mean'])**2 / (2*gt[k]['stdv']**2)))
            n_scored = n_scored + 1
        else:
            print(f"Warning (from user): Labeled instance '{k}' not present in the ground truth!")

    # If num. scored instances != number of ground truth instances, warn user
    if n_scored < len(gt):
        print(f"Warning (from user): Only {n_scored}/{len(gt)} instances were evaluated. " +
              f"Your maximum possible score is {100 * (n_scored / len(gt))}% of total.")

    # Return score
    return score / len(gt)


def main():
    if len(sys.argv) < 2:
        raise ValueError('Output file name must be specified!')

    # Load ground truth & predictions
    labels = load_labels('gt/test_gt.csv')
    preds = load_predictions(sys.argv[1])

    # Evaluate predictions
    score = evaluate_predictions(labels, preds)
    print('-' * 50)
    print(f'Score: {score}')
    print('-' * 50)


if __name__ == '__main__':
    main()
