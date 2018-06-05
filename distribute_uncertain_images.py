import shutil

LABELS_TO_DESTDIRS = {
    'c': 'ChaLearn/ethnicity/caucasian',
    'b': 'ChaLearn/ethnicity/black',
    'a': 'ChaLearn/ethnicity/asian',
    'i': 'ChaLearn/ethnicity/indian',
    'o': 'ChaLearn/ethnicity/others',
}


if __name__ == '__main__':
    with open('chalearn_uncertain.txt', 'r') as f:
        for line in f.readlines():
            file_path, label = line.split(',')
            shutil.copy2(file_path, LABELS_TO_DESTDIRS[label[0]])
