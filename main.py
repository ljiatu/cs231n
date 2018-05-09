from dataset import IMDbFacialDataset
import matplotlib.pyplot as plt


def main():
    dataset = IMDbFacialDataset('imdb_crop')
    print(0, dataset[0].shape)
    print(460722, dataset[460722].shape)
    print(40302, dataset[40302].shape)


if __name__ == '__main__':
    main()
