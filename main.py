from dataset import IMDbFacialDataset
import matplotlib.pyplot as plt


def main():
    dataset = IMDbFacialDataset('imdb_crop')
    for i in range(len(dataset)):
        print(i, dataset[i].shape)

        if i == 3:
            plt.show()
            break


if __name__ == '__main__':
    main()
